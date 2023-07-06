# 2DMatProcess
This is a group of tools for processing micrographs of 2D materials on various substrates

## Wrinkle Detect
Detects and quantifies wrinkles, designed for analyzing graphene on Nafion.  Outputs a CSV containing the heights and widths of every detected wrinkle, along with the total number of peaks and valleys detected.

### Example
Input AFM image of graphen flakes on Nafion
![AFM Image](./2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etoh-16gray.png?raw=true)

Input "mask", manually created to indicate locations for analysis (where the graphene is)
![AFM Image Mask](./2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etoh-16gray-flakes.png?raw=true)

Output wrinkle locations (peaks are indicated in green, valleys are indicated in blue)
![Wrinkle Locations](./2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etohwrinkles.png?raw=true)

The program outputs a [CSV](./2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etoh.csv)
## License
[MIT](https://choosealicense.com/licenses/mit/)
