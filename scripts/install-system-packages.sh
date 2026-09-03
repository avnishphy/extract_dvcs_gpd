#!/bin/sh
set -eu

snapshot="${1:?Ubuntu snapshot timestamp is required}"
shift
if [ "$#" -eq 0 ]; then
    echo "system package list is empty" >&2
    exit 64
fi

# snapshot.ubuntu.com occasionally returns short-lived 5xx responses. Keep the
# retry window bounded: install.sh resumes a completed SIF rather than starting
# another expensive build, and a first build should fail clearly during an
# extended outage instead of appearing stuck for hours.
apt_config_dir="${DVCS_APT_CONFIG_DIR:-/etc/apt/apt.conf.d}"
apt_command="${DVCS_APT_COMMAND:-apt-get}"
retry_attempts="${DVCS_APT_RETRY_ATTEMPTS:-3}"
retry_delay_base="${DVCS_APT_RETRY_DELAY_BASE:-5}"
object_retries="${DVCS_APT_OBJECT_RETRIES:-3}"
network_timeout="${DVCS_APT_NETWORK_TIMEOUT:-45}"
case "${retry_attempts}:${retry_delay_base}:${object_retries}:${network_timeout}" in
    *[!0-9:]*|0:*|*:) echo "invalid APT retry configuration" >&2; exit 64 ;;
esac
mkdir -p "${apt_config_dir}"
cat > "${apt_config_dir}/80dvcs-network-retries" <<EOF
Acquire::Retries "${object_retries}";
Acquire::http::Timeout "${network_timeout}";
Acquire::https::Timeout "${network_timeout}";
EOF

apt_retry() {
    attempt=1
    while [ "${attempt}" -le "${retry_attempts}" ]; do
        if "${apt_command}" "$@"; then
            return 0
        fi
        if [ "${attempt}" -eq "${retry_attempts}" ]; then
            echo "apt-get failed after ${attempt} attempts: $*" >&2
            return 1
        fi
        delay=$((attempt * retry_delay_base))
        echo "apt-get attempt ${attempt}/${retry_attempts} failed; retrying in ${delay}s: $*" >&2
        sleep "${delay}"
        attempt=$((attempt + 1))
    done
}

# The pinned minimal base has no CA bundle. Bootstrap only that package from
# the signed Ubuntu archive, then route every scientific runtime package
# through the immutable snapshot.
apt_retry update
apt_retry install -y --no-install-recommends ca-certificates
printf 'APT::Snapshot "%s";\n' "${snapshot}" > "${apt_config_dir}/50snapshot"
apt_retry update
apt_retry install -y --no-install-recommends "$@"
