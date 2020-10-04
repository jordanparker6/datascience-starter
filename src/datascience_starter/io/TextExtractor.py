import os, string, argparse
from tika import parser
from base.logging import Logger

class TextExtactor(Logger):
    """
    TextExtractor
     Sends requests to an Apache Tika to extract text from all thefiles within a directory.
     * Please note that java_runtime_eviroment and tesseract must be installed and with
       the PATH varriable for the Tika server to work.
     args:
       -> output_dir: the directory to save extracted text files
    """
    def __init__(self, output_dir):
        super().__init__()
        self.server = 'http://localhost:9998/rmeta/text'
        self.headers = { 'X-Tika-PDFextractInlineImages': 'true' }
        self.output_dir = output_dir

    def extract(self, input_dir, limit=None):
        """
        Walks through each file in the direcotry and complete the following:
         (1) rename the filename to ascii (Apache Tika requirement);
         (2) parse the file to .txt with Apache Tika;
         (3) encode the text to 'utf-8', ignoring encoding errors;
         (4) save the bytes to the output directory
        args:
          -> input_dir: the directory to crawl and extract
          -> limit: an integer limit on number of files crawled
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

    def _rename_ascii(self, root, file):
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