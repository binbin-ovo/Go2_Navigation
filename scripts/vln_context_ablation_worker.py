"""Inference-only offline ablation worker, used only in coordinated Stage B."""

import json
from pathlib import Path
import sys
import time

from go2_perception.vln.context_ablation import sample_factory, validate_contract, verify_native_layout
from vln_shadow_worker import main


def serve(agent,args,send,writer,run_job,validate_jobs):
    for line in sys.stdin:
        if writer is not None:
            writer.check()
        message=json.loads(line)
        if set(message)!={"identity","job","destination","ablation"}:
            raise ValueError("Unsupported offline ablation envelope")
        job,spec=message["job"],message["ablation"]
        validate_jobs({"schema":"lightnav-local-jobs-v1","jobs":[job]})
        validate_contract(spec,job)
        verify_native_layout(spec,agent.engine.bundle)
        begin=time.monotonic()
        result=run_job(agent,job,Path(message["destination"]),args.device,
                       profile_stages=args.profile_stages,evidence_writer=writer,sample_factory=sample_factory(spec))
        end=time.monotonic()
        send({"identity":message["identity"],"result":result,"host_inference_start_s":begin,
              "host_inference_end_s":end,"inference_duration_s":end-begin,"backend":"lightnav"})


if __name__=="__main__":
    main(serve_fn=serve)
