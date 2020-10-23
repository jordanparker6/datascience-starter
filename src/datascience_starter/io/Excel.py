import xlwings as xw
import pandas as pd
from datascience_starter.base.logging import Logger

class Excel(Logger):
    """
    Excel
     An Excel workbook utility class
      args:
       -> file: Excel workbook file name
    """
    def __init__(self, file):
        super().__init__()
        self.file_name = file
        self.wb = xw.Book(file)
        self.log.info(f"Excel File Read: {file}")

    def to_df(self, sheet, range="A1", expand='table'):
        df = pd.DataFrame(self.wb.sheets(sheet).range(range).expand(expand).value) 
        headers = df.iloc[0]
        df = pd.DataFrame(df.values[1:], columns=headers)
        return df

    def to_excel(self, df, sheet, range="A1", clear_range=None):
        if clear_range is not None:
            self.wb.sheets(sheet).range(clear_range).clear_contents()
        print_range = wb.sheets(sheet).range(range) 
        print_range.value = df