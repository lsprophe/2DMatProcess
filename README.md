# 2DMatProcess
This is a group of tools for processing micrographs of 2D materials on various substrates

## Wrinkle Detect
Detects and quantifies wrinkles, designed for analyzing graphene on Nafion.  Outputs a CSV containing the heights and widths of every detected wrinkle, along with the total number of peaks and valleys detected.

### Example
Input AFM image of graphen flakes on Nafion
![AFM Image](https://github.com/lsprophe/2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etoh-16gray.png)

Input "mask", manually created to indicate locations for analysis (where the graphene is)
![AFM Image Mask](https://github.com/lsprophe/2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etoh-16gray-flakes.png)

Output wrinkle locations (peaks are indicated in green, valleys are indicated in blue)
![Wrinkle Locations](https://github.com/lsprophe/2DMatProcess/WrinkleDetect/examples/np2_3-spincoat-n117-75etohwrinkles.png)

The program outputs a CSV
## License
[MIT](https://choosealicense.com/licenses/mit/)
