# Data analysis project

Our project is titled **Inflation and the consequences on key factors in the American economy** and is about the relationship between unemployment and inflation and how growth in GDP is very tightly correlated to inflation.

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets** which have been collected using API from the World bank:

1. GDP growth (annual %) - United States (*API: NY.GDP.MKTP.KD.ZG*)
1. Inflation, consumer prices (annual %) - United States (*API: FP.CPI.TOTL.ZG*)
1. Employment to population ratio, 15+, total (%) (national estimate) - United States (*API: SL.EMP.TOTL.SP.NE.ZS*)
1. Real interest rate (%) - United States (*API: FR.INR.RINR*)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

``pip install matplotlib-venn``
``pip install git+https://github.com/alemartinello/dstapi``
``pip install pandas-datareader``