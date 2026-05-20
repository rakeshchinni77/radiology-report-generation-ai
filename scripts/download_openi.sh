#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/raw"
IMAGES_DIR="${DATA_DIR}/images"
XML_DIR="${DATA_DIR}/xml"
REPORTS_DIR="${DATA_DIR}/reports"

DOWNLOAD_URL=""
ARCHIVE_PATH=""
EXTRACT_DIR="${DATA_DIR}"

log() {
	printf '[download_openi] %s\n' "$1"
}

fail() {
	printf '[download_openi][error] %s\n' "$1" >&2
	exit 1
}

usage() {
	cat <<'EOF'
Usage:
	scripts/download_openi.sh [--archive PATH] [--url URL] [--extract-dir PATH]

Examples:
	scripts/download_openi.sh --archive ~/Downloads/openi.zip
	scripts/download_openi.sh --url https://example.com/openi.zip

The script never deletes existing files. It only creates missing directories,
extracts archives into the chosen location, and verifies that OpenI-style files
exist under data/raw/images, data/raw/xml, and data/raw/reports.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--archive)
			ARCHIVE_PATH="${2:-}"
			shift 2
			;;
		--url)
			DOWNLOAD_URL="${2:-}"
			shift 2
			;;
		--extract-dir)
			EXTRACT_DIR="${2:-}"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			fail "Unknown argument: $1"
			;;
	esac
done

mkdir -p "${IMAGES_DIR}" "${XML_DIR}" "${REPORTS_DIR}"

log "Dataset directories are ready"
log "Images: ${IMAGES_DIR}"
log "XML: ${XML_DIR}"
log "Reports: ${REPORTS_DIR}"

if [[ -n "${DOWNLOAD_URL}" && -z "${ARCHIVE_PATH}" ]]; then
	url_name="${DOWNLOAD_URL%%\?*}"
	url_name="${url_name##*/}"
	if [[ "${url_name}" == *.* ]]; then
		ARCHIVE_PATH="${ROOT_DIR}/data/${url_name}"
	else
		ARCHIVE_PATH="${ROOT_DIR}/data/openi_download.zip"
	fi
	log "Downloading archive to ${ARCHIVE_PATH}"

	if command -v curl >/dev/null 2>&1; then
		curl -L --fail --retry 3 --retry-delay 2 --output "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
	elif command -v wget >/dev/null 2>&1; then
		wget -O "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
	else
		fail "curl or wget is required for automatic download"
	fi
fi

if [[ -n "${ARCHIVE_PATH}" ]]; then
	[[ -f "${ARCHIVE_PATH}" ]] || fail "Archive not found: ${ARCHIVE_PATH}"

	log "Extracting archive from ${ARCHIVE_PATH} into ${EXTRACT_DIR}"
	mkdir -p "${EXTRACT_DIR}"

	case "${ARCHIVE_PATH}" in
		*.zip)
			if command -v unzip >/dev/null 2>&1; then
				unzip -o "${ARCHIVE_PATH}" -d "${EXTRACT_DIR}"
			else
				fail "unzip is required for .zip archives"
			fi
			;;
		*.tar.gz|*.tgz)
			tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"
			;;
		*.tar)
			tar -xf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"
			;;
		*)
			fail "Unsupported archive format: ${ARCHIVE_PATH}"
			;;
	esac
fi

image_count=$(find "${IMAGES_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.dcm' \) | wc -l | tr -d ' ')
xml_count=$(find "${XML_DIR}" -type f -iname '*.xml' | wc -l | tr -d ' ')
report_count=$(find "${REPORTS_DIR}" -type f \( -iname '*.txt' -o -iname '*.csv' -o -iname '*.json' \) | wc -l | tr -d ' ')

log "Verification summary"
log "Image files found: ${image_count}"
log "XML files found: ${xml_count}"
log "Report files found: ${report_count}"

if [[ "${image_count}" -eq 0 && "${xml_count}" -eq 0 && "${report_count}" -eq 0 ]]; then
	fail "No OpenI files were found. Check that the archive was extracted into the expected folders."
fi

log "OpenI dataset acquisition setup completed successfully"
