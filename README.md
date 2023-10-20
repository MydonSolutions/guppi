# Guppi

Simplified handling of GUPPI RAW files.

Makes use of [rao_keyvalue_property_mixin_classes](https://github.com/MydonSolutions/rao_keyvalue_property_mixin_classes) to implement GuppiRawHeader classes that are have
properties specific to RAO GUPPI RAW outputs. Furthermore, the appropriate class is
automatically inferred for each header read from file, based on the `.telescope` property
(`TELESCOP` keyed-value).

## Examples

<details><summary>Reading</summary>
To read all the blocks of a single GUPPI RAW file:

```
from guppi import GuppiHandler

for gr_header, gr_data in GuppiHandler("some_guppi_file.0000.raw").blocks():
		pass
```

The `GuppiHandler` also manages multi-file GUPPI RAW sets. Either pass it the complete
set of filepaths in the order to process, or pass it the stem:

```
from guppi import GuppiHandler
import glob

guppi_filepaths = glob.glob("some_multi_guppi_file.*.raw")
guppi_filepaths.sort()

for gr_header, gr_data in GuppiHandler(guppi_filepaths).blocks():
		pass

# or, equivalently:

for gr_header, gr_data in GuppiHandler("some_multi_guppi_file.").blocks():
		pass
```

</details>

<details><summary>Writing</summary>
A single, static, function can be called reiteratively with to write a GuppiRawHeader and `numpy.array` as a block.

```
from guppi import GuppiHandler

for i in range(3):
		GuppiHandler.write_to_file(
				"output_path.0000.raw",
				gr_header,
				gr_data,
				file_open_mode=('wb' if i == 0 else 'ab')
		)
```

</details>

## Installation:
A `pip install .` after cloning the repository is sufficient to install the library
