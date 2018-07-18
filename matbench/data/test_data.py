# coding: utf-8

import pandas as pd
import unittest

import matbench.data.load as loaders_source
import matminer.datasets.dataframe_loader as blacklist_source


class MatbenchTestData(unittest.TestCase):

    def test_load_data(self):
        loader_func_names = []
        for f in dir(loaders_source):
            if 'load_' in f and f not in dir(blacklist_source):
                loader_func_names.append(f)

        print(loader_func_names)
        for loader_func_name in loader_func_names:
            print('testing "{}()" ...'.format(loader_func_name))
            loader_func = getattr(loaders_source, loader_func_name)
            df = loader_func()
            loader_func_doc = loader_func.__doc__
            self.assertTrue(isinstance(df, pd.DataFrame))
            for field in ['References:', 'Returns:', '(target):', '(input):']:
                self.assertTrue(field in loader_func_doc)
            for col in df:
                self.assertEqual(col, col.lower())
                self.assertLessEqual(len(col), 20)


if __name__ == '__main__':
    unittest.main()