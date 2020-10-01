import xlwings as xw
import pandas as pd
from base.logging import Logger

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

    def to_df(self, sheet, range, expand='table'):
        df = pd.DataFrame(self.wb.sheets(sheet).range(range).expand(expand).value) 
        headers = df.iloc[0]
        df = pd.DataFrame(df.values[1:], columns=headers)
        return df

    def to_excel(self, df, sheet, print_range, clear_range=None):
        self.wb(self.file_name) 
        if clear_range is not None:
            self.wb.sheets(sheet).range(clear_range).clear_contents()
        print_range = wb.sheets(sheet).range(print_range) 
        print_range.value = df