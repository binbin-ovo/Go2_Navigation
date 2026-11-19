#!/bin/bash
# Published operator command; the shared launcher validates every live opt-in.
exec bash "$(dirname "$0")/../hardware/launch_supervised.sh" one-action "$@"
