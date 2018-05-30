# coding: utf-8

import pandas as pd
import unittest

import matbench.data.load as loaders_source


class MatbenchTestData(unittest.TestCase):

    def test_load_data(self):
        loader_funcs = [f for f in dir(loaders_source) if 'load_' in f]
        print(loader_funcs)
        for loader_func in loader_funcs:
            print('testing "{}()" ...'.format(loader_func))
            df = getattr(loaders_source, loader_func)()
            self.assertTrue(isinstance(df, pd.DataFrame))
            for col in df:
                self.assertEqual(col, col.lower())
                self.assertLessEqual(len(col), 20)


if __name__ == '__main__':
    unittest.main()