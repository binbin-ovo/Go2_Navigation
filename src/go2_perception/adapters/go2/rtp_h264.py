"""Passive H.264-over-RTP depacketizing (RFC 6184 single NAL, STAP-A, FU-A) and decoding."""

from dataclasses import dataclass

START_CODE = b"\0\0\0\1"
RTP_CLOCK_HZ = 90000


@dataclass
class RtpPacket:
    seq: int
    timestamp: int
    marker: bool
    ssrc: int
    payload_type: int
    payload: bytes


def parse_rtp(packet):
    """Return an RtpPacket, or None for anything that is not well-formed RTP v2."""
    if len(packet) < 12 or packet[0] >> 6 != 2:
        return None
    offset = 12 + 4 * (packet[0] & 0x0F)
    if packet[0] & 0x10:
        if len(packet) < offset + 4:
            return None
        offset += 4 + 4 * int.from_bytes(packet[offset + 2 : offset + 4], "big")
    end = len(packet)
    if packet[0] & 0x20:
        end -= packet[-1]
    if offset >= end:
        return None
    return RtpPacket(
        int.from_bytes(packet[2:4], "big"),
        int.from_bytes(packet[4:8], "big"),
        bool(packet[1] & 0x80),
        int.from_bytes(packet[8:12], "big"),
        packet[1] & 0x7F,
        bytes(packet[offset:end]),
    )


@dataclass
class AccessUnit:
    rtp_timestamp: int
    host_recv_first: float
    host_recv_last: float
    data: bytes  # Annex-B
    complete: bool  # False when a packet of this unit was lost or reordered
    keyframe: bool
    parameter_sets_only: bool = False


class AccessUnitAssembler:
    """Groups RTP packets into access units, tracking packet loss by sequence number."""

    def __init__(self):
        self.packets = 0
        self.lost_packets = 0
        self.reordered_packets = 0
        self.ignored_packets = 0
        self._last_seq = None
        self._reset()

    def _reset(self):
        self._ts = None
        self._first = self._last = 0.0
        self._data = bytearray()
        self._complete = True
        self._keyframe = False
        self._in_fragment = False
        self._non_parameter_nal = False

    def _flush(self):
        if self._ts is None or not self._data:
            self._reset()
            return []
        unit = AccessUnit(
            self._ts,
            self._first,
            self._last,
            bytes(self._data),
            self._complete and not self._in_fragment,
            self._keyframe,
            not self._non_parameter_nal,
        )
        self._reset()
        return [unit]

    def feed(self, packet, host_recv):
        """Add a packet; return the access units that it completed (zero or more)."""
        out = []
        self.packets += 1
        lost = False
        if self._last_seq is not None:
            delta = (packet.seq - self._last_seq) & 0xFFFF
            if delta == 0 or delta >= 0x8000:
                self.reordered_packets += 1
                self._complete = False
                return out
            if delta > 1:
                self.lost_packets += delta - 1
                lost = True
        self._last_seq = packet.seq
        if lost:  # the missing packets may belong to the open unit or to this new one
            self._complete = False
        if self._ts is not None and packet.timestamp != self._ts:
            out += self._flush()
        if self._ts is None:
            self._ts, self._first = packet.timestamp, host_recv
        if lost:
            self._complete = False
        self._last = host_recv
        self._add_payload(packet.payload)
        if packet.marker:
            out += self._flush()
        return out

    def _add_nal(self, nal):
        nal_type = nal[0] & 0x1F
        if nal_type == 5:
            self._keyframe = True
        if nal_type not in (7, 8):
            self._non_parameter_nal = True
        self._data += START_CODE + nal

    def _add_payload(self, payload):
        nal_type = payload[0] & 0x1F
        if 1 <= nal_type <= 23:
            self._add_nal(payload)
        elif nal_type == 24:  # STAP-A
            i = 1
            while i + 2 <= len(payload):
                size = int.from_bytes(payload[i : i + 2], "big")
                i += 2
                if size == 0 or i + size > len(payload):
                    self._complete = False
                    break
                self._add_nal(payload[i : i + size])
                i += size
        elif nal_type == 28 and len(payload) >= 2:  # FU-A
            start, end = bool(payload[1] & 0x80), bool(payload[1] & 0x40)
            if start:
                if self._in_fragment:
                    self._complete = False
                header = bytes([(payload[0] & 0xE0) | (payload[1] & 0x1F)])
                self._add_nal(header + payload[2:])
                self._in_fragment = not end
            elif self._in_fragment:
                self._data += payload[2:]
                self._in_fragment = not end
            else:
                self._complete = False  # fragment whose start was lost
        else:
            self.ignored_packets += 1


class H264Decoder:
    """PyAV decoder returning (BGR image, rtp_timestamp) with the capture stamp preserved."""

    def __init__(self):
        import av

        self._av = av
        self._ctx = av.CodecContext.create("h264", "r")
        self._ctx.thread_type = "NONE"  # frame threading would add latency
        self.errors = 0

    def decode(self, unit):
        packet = self._av.Packet(unit.data)
        packet.pts = unit.rtp_timestamp
        try:
            frames = self._ctx.decode(packet)
        except self._av.error.InvalidDataError:
            self.errors += 1
            return []
        return [(f.to_ndarray(format="bgr24"), f.pts) for f in frames]


def unwrap_rtp(previous_unwrapped, timestamp):
    """Extend a 32-bit RTP timestamp monotonically; pass None for the first sample."""
    if previous_unwrapped is None:
        return int(timestamp)
    delta = (int(timestamp) - previous_unwrapped) & 0xFFFFFFFF
    if delta >= 0x80000000:
        delta -= 0x100000000
    return previous_unwrapped + delta


def packetize_annexb(data, timestamp, first_seq=0, mtu=1200, ssrc=1, payload_type=96):
    """Inverse of the assembler for offline tests: Annex-B access unit -> RTP datagrams."""
    nals = [n for n in bytes(data).replace(b"\0\0\0\1", b"\0\0\1").split(b"\0\0\1") if n]
    payloads = []
    for nal in nals:
        if len(nal) <= mtu:
            payloads.append(nal)
            continue
        indicator = (nal[0] & 0xE0) | 28
        body = nal[1:]
        chunks = [body[i : i + mtu] for i in range(0, len(body), mtu)]
        for i, chunk in enumerate(chunks):
            flags = (nal[0] & 0x1F) | (0x80 if i == 0 else 0)
            flags |= 0x40 if i == len(chunks) - 1 else 0
            payloads.append(bytes([indicator, flags]) + chunk)
    packets = []
    for i, payload in enumerate(payloads):
        marker = 0x80 if i == len(payloads) - 1 else 0
        header = bytes([0x80, marker | payload_type])
        header += ((first_seq + i) & 0xFFFF).to_bytes(2, "big")
        header += (timestamp & 0xFFFFFFFF).to_bytes(4, "big") + ssrc.to_bytes(4, "big")
        packets.append(header + payload)
    return packets
