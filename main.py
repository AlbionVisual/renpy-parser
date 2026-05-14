from unpack_archives import runArchives
from unpack_renpy import unpackRenpy
from parse_rpy import runParse
import argparse
from consts import DOWNLOADS_DEFAULT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--downloads", default=DOWNLOADS_DEFAULT)
    parser.add_argument("--no-file-filter", action="store_true")
    args = parser.parse_args()

    code, archive_mapping = runArchives(args)
    if code == 0:
        unpackRenpy(args.no_file_filter)
        code = runParse(args, archive_mapping)

    exit(code)
