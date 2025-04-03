#!/bin/bash
set -euo pipefail
ROOT_DIR=$( (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) )

temp="$(mktemp -d)"
output="$ROOT_DIR"
index_filename="index.qmd"

(
  cd ../arXiv-2501.14787v1 || exit 1
  set --
  set -- "$@" --wrap=preserve
  set -- "$@" --standalone
  set -- "$@" -Mlang=en-US
  set -- "$@" --embed-resources=true
#  set -- "$@" -t markdown
#  set -- "$@" -t markdown_mmd
  set -- "$@" -t commonmark_x
  set -- "$@" -o "$temp/$index_filename" --extract-media="$temp"
  pandoc main.tex "$@" || exit 1
  # Add '{% raw %}' after frontmatter and '{% endraw %}' at the end
  perl -i -pe 'if (/^---$/) { $count++; $_ .= "\n{% raw %}" if $count==2 }' "$temp/$index_filename"
  #sed -i -e '1s/^/{% raw %}\n/' "$temp/$index_filename"
  sed -i -e '$s/$/\n{% endraw %}/' "$temp/$index_filename"
) || exit 1

convert_asset() {
  local input="$1"
  local md="$2"
  local filename="$(basename "$input")"
  local relative="$(dirname "${input#$temp/}")"
  local dest="$output/$relative"
  local output
  mkdir -p "$dest"

  local ext="${input##*.}"
  case "$ext" in
    pdf)
      output="$dest/$filename.svg"
      pdf2svg "$input" "$output"
      #inkscape "$input" --export-type=svg --export-filename="$output"
      ;;
    jpg|jpeg|png|gif)
      output="$dest/$filename.webp"
      convert "$input" "$output"
      ;;
    *)
      output="$dest/$filename"
      cp "$input" "$output"
      ;;
  esac

  escaped_input=$(printf '%s\n' "$input" | sed -e 's/[\/&]/\\&/g')
  rel_out=$(realpath --relative-to="${md%/*}" "$output")
  sed -i -e "s|$escaped_input|$rel_out|g" "$md"
}

export -f convert_asset
export output temp

cp "$temp/$index_filename" "$output/$index_filename" || exit 1
find "$temp" -type f -not -name "$index_filename" -exec bash -c 'convert_asset "$0" "$1"' {} "$output/$index_filename" ';'
#find "$temp" -type f -not -name "$index_filename" \
#  | parallel --no-notice --jobs 4 convert_asset {} "$output/$index_filename"
#find "$temp" -type f -not -name "$index_filename" -print0 \
#  | xargs -0 -n 1 -P 4 bash -c 'convert_asset "$0" "$1"' {} "$output/$index_filename"

rm -rf "$temp"
