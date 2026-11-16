"""Offline experiment worker; reuses the fixed shadow model/runtime bootstrap."""

import json
from pathlib import Path
import sys
import time

from go2_perception.vln.bounded_context import sample_factory, validate_context
from vln_shadow_worker import main


def serve(agent, args, send, writer, run_job, validate_jobs):
    for line in sys.stdin:
        if writer is not None:
            writer.check()
        message = json.loads(line)
        if set(message) != {"identity", "job", "destination", "bounded_context"}:
            raise ValueError("Unsupported offline comparison envelope")
        job, context = message["job"], message["bounded_context"]
        validate_jobs({"schema": "lightnav-local-jobs-v1", "jobs": [job]})
        if context is not None:
            validate_context(context, job)
        start = time.monotonic()
        result = run_job(agent, job, Path(message["destination"]), args.device,
                         profile_stages=args.profile_stages, evidence_writer=writer,
                         sample_factory=None if context is None else sample_factory(context))
        end = time.monotonic()
        send({"identity": message["identity"], "result": result,
              "host_inference_start_s": start, "host_inference_end_s": end,
              "inference_duration_s": end - start, "backend": "lightnav"})


if __name__ == "__main__":
    main(serve_fn=serve)
