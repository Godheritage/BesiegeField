PROJ_ROOT="$(dirname "$(realpath "$0")")"
CONFIG_PATH="${PROJ_ROOT}"
CONFIG_NAME="rl_config"
# ----------------------------

YAML_FILE="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

DEFAULT_LOCAL_DIR=$(yq -r '.trainer.default_local_dir // empty' "$YAML_FILE")

if [[ -n "${PROJ_ROOT}/$DEFAULT_LOCAL_DIR" ]]; then
    mkdir -p "${PROJ_ROOT}/$DEFAULT_LOCAL_DIR"
    cp "$YAML_FILE" "${PROJ_ROOT}/${DEFAULT_LOCAL_DIR}/${CONFIG_NAME}.yaml"
fi

python -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME"
