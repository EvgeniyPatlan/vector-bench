"""MariaDB 12.3 MHNSW, driven by ann-benchmarks.

Identical to the 11.8 module in every respect except the name. It exists as a
separate algorithm because ann-benchmarks keys its result files on the
algorithm name and its index parameters, with no notion of which build
produced them. Sharing the name with `mariadb` would make a 12.3 run find
11.8.8's results, report "nothing to run", and return the old numbers in
seconds -- the failure this project has already been bitten by twice.

The dialect is imported rather than copied so the two versions cannot drift
apart in how they are configured, which would turn a version comparison into a
configuration comparison.
"""

from ..mariadb.module import MARIADB
from ...vb_mysql import VBMySQLBase


class MariaDB123(VBMySQLBase):
    dialect = MARIADB
