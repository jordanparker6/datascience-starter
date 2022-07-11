import string
import argparse
import hashlib
import datetime as dt
import logging
from pathlib import Path
from tika import parser
from typing import Optional, List, Generator

log = logging.getLogger(__name__)

class TextExtactor:
    """ A class to extract text from files within a directoring using Apache Tika.
     
    Please note that java_runtime_eviroment and tesseract must be installed and with
    the PATH varriable for the Tika server to work.
     
    Args:
        tika_server (str): The hostname and port of the tika server.
        output_dir (str): The directory to save extracted text files.

    Attributes:
        ouput_dir (Path): The directory to write output files.
    """
    def __init__(self, tika_server: str, output_dir: str):
        super().__init__()
        self._server = tika_server
        self._headers = { 
                'X-Tika-PDFextractInlineImages': 'true',
                "X-Tika-OCRLanguage": "eng",
                "X-Tika-OCRTimeout": "300"
            }
        self._tika_timeout = 300
        self.output_dir = Path(output_dir)

    def extract(self,
            input_dir: Path, 
            file_types: Optional[List[str]] = None, 
            limit: Optional[int] = None
        ) -> Generator:
        """ Extracts the text from each file and writes to the output directory.
        Args:
            input_dir: The directory to crawl and extract text.
            limit: The limit on the number of files extracted.

        Yields:
            The metadata and extracted text.
        """
        files = self._find_files(input_dir, file_types)
        for i, path in enumerate(files):
            _id = self.hash_file(path)
            if self.check_duplicate(_id):
                log.debug("Duplicate: File found in corpus.") # what if path changed
                continue
            else:
                log.info('Extracting: {}'.format(path))
                doc = parser.from_file(
                        str(path), 
                        serverEndpoint=f'{self._server}/rmeta/text', 
                        requestOptions={
                            "headers": self._headers,
                            "timeout": self._tika_timeout
                        }
                    )
                if doc['status'] == 200:
                    metadata = doc['metadata']
                    self._clean_metadata(metadata)  # add more metadata to extract
                    text = str(doc['content']).strip()
                    btext = text.encode('windows-1252', errors='ignore')  # clean up utf-8 errors
                    self.output_dir.joinpath(f"{_id}.txt").write_bytes(btext)
                    meta = {
                        "corpus_id": _id,
                        "path": str(path.absolute()),
                        "extension": path.suffix,
                        "extract_time": dt.datetime.now()
                    }
                    yield meta, text
                else:
                    log.error('Unsucessfull request {}.'.format(doc['status']))
            
            if limit is not None and i >= limit:
                break

    def hash_file(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def check_duplicate(self, hash: str) -> bool:
        for fname in self.output_dir.iterdir():
            if hash == fname.stem:
                return True
        return False

    def _find_files(self, input_dir: Path, ftype_rstrict: Optional[List[str]] = None) -> List[Path]:
        if ftype_rstrict:
            files = []
            for ftype in ftype_rstrict:
                files.extend([f for f in Path(input_dir).glob("**/*" + ftype) if f.is_file()])
            return files
        else:
            return [f for f in Path(input_dir).glob("**/*") if f.is_file()]

    def _clean_metadata(self, metadata):
        return metadata

if __name__ == "_main_":
    flags = argparse.ArgumentParser()
    flags.add_argument('--input', help='Root folder of the documents to parse.')
    flags.add_argument('--output', help='Output folder of the documents parsed.')
    flags.add_argument('-n', '--number', help='Number of files in root to be parsed.', default='all')
    args = flags.parse_args()

    extractor = TextExtactor(args.output)
    extractor.extract(args.input, args.number)