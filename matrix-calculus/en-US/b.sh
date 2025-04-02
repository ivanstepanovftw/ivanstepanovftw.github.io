#!/bin/bash
set -euo pipefail
ROOT_DIR=$( (cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P) )

temp="$(mktemp -d)"
output="$ROOT_DIR"

(
  cd ../arXiv-2501.14787v1 || exit 1
  pandoc main.tex --wrap=preserve --standalone -Mlang=en-US --embed-resources=true -t markdown -o "$temp/index.md" --extract-media="$temp"
  # Add {% raw %} after frontmatter and {% endraw %} at the end
  #perl -i -pe 'if(/^---$/){ $count++; if($count==2){ print "{% raw %}\n" } }' "$temp/index.md"
   perl -i -pe 'if (/^---$/) { $count++; $_ .= "\n{% raw %}" if $count==2 }' "$temp/index.md"
  #sed -i -e '1s/^/{% raw %}\n/' "$temp/index.md"
  sed -i -e '$s/$/\n{% endraw %}/' "$temp/index.md"
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

cp "$temp/index.md" "$output/index.md" || exit 1
find "$temp" -type f -not -name 'index.md' -exec bash -c 'convert_asset "$0" "$1"' {} "$output/index.md" ';'
#find "$temp" -type f -not -name 'index.md' \
#  | parallel --no-notice --jobs 4 convert_asset {} "$output/index.md"
#find "$temp" -type f -not -name 'index.md' -print0 \
#  | xargs -0 -n 1 -P 4 bash -c 'convert_asset "$0" "$1"' {} "$output/index.md"

echo rm -rf "$temp"
