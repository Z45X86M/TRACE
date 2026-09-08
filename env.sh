# Source this before running any script:  source env.sh
# All release code imports flat module names; PYTHONPATH resolves them across src/ subdirs.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$_ROOT/src/extraction:$_ROOT/src/common:$_ROOT/src/retrieval:$_ROOT/src/verify:$_ROOT/src/baselines:$_ROOT/src/bench:$PYTHONPATH"
export TRACE_ROOT="$_ROOT"
