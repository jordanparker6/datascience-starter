import os, string, argparse
from tika import parser
from typing import Union
from datascience_starter.base.logging import Logger

class TextExtactor(Logger):
    """ A class to extract text from files within a directoring using Apache Tika.
     
    Please note that java_runtime_eviroment and tesseract must be installed and with
    the PATH varriable for the Tika server to work.
     
    Args:
        output_dir (str): The directory to save extracted text files.

    Attributes:
        server (str): The server address.
        headers (dict): The headers for the Apache Tika requests.
        ouput_dir (str): The directory to write output files.

    """
    def __init__(self, output_dir: str):
        super().__init__()
        self.server = 'http://localhost:9998/rmeta/text'
        self.headers = { 'X-Tika-PDFextractInlineImages': 'true' }
        self.output_dir = output_dir

    def extract(self, input_dir: str, limit: Union[int, None] = None):
        """ Extracts the text from each file and writes to the output directory.

        Args:
            input_dir: The directory to crawl and extract text.
            limit: The limit on the number of files extracted.

        """

        for i, paths in enumerate(os.walk(input_dir)):
            root, dirs, files = paths
            for name in files:
                name = self._rename_ascii(root, name)
                file_name = os.path.join(root, name)
                self.log.info('Extracting -- {}'.format(name))
                doc = parser.from_file(file_name, serverEndpoint=self.server, headers=self.headers)
                if doc['status'] == 200:
                    text = doc['content']
                    text = str(text)
                    safe_text = text.encode('utf-8', errors='ignore')
                    corpus_fn = '{} -- {}.txt'.format(i, name)
                    with open(self.output_dir + corpus_fn, 'wb') as f:
                        f.write(safe_text.strip())
                if limit is None:
                    continue
                elif i >= int(limit):
                    break
            else:
                continue
            break

    def _rename_ascii(self, root: str, file: str) -> str:
        """Renames file names so that they are ascii compliant.

        Args:
            root: The root of the file path.
            file: The file name

        Returns:
            An ascii compliant file name.

        """
        try:
            file.encode('ascii')
            file_name = file
        except UnicodeEncodeError:
            new_file = ''.join(c for c in file if c in string.printable)
            file_name = new_file
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
        return file_name


if __name__ == "_main_":
    flags = argparse.ArgumentParser()
    flags.add_argument('--input', help='Root folder of the documents to parse.')
    flags.add_argument('--output', help='Output folder of the documents parsed.')
    flags.add_argument('-n', '--number', help='Number of files in root to be parsed.', default='all')
    args = flags.parse_args()

    extractor = TextExtactor(args.output)
    extractor.extract(args.input, args.number)