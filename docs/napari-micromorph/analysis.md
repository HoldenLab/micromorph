# Analysing an Image
You can choose analysis parameters by clicking on Analysis Settings.
Here's a brief description of what each parameters does:

- Pixel Size: the pixel size of the image you are analysing, in nm/px!
- PSF FWHM: full-width at half-maximum of the psf, in nm. Only used if Fit Type is "fluorescence"
- Fit Type: choose between `fluorescence`, `phase` and `tophat`. Use `fluorescence` for membrane staining and `tophat` for cytoplasmic fluorescence.

Full Analysis Parameters

- Number of Widths: number of times the width is sampled over the cell.
- Boundary Smoothing Factor: we filter out high frequency data in the cell boundary by applying a Fourier Transform and keeping only the frequencies equivalent to the number set here. Higher number = more pixelated boundary, lower number = higher smoothing.
- Error Threshold, Max Iterations, Step Size, Spline Spacing, Spline Order: these are all defined in px, and are the parameters used to extended the medial axis to the cell boundary and smooth it. A higher value of error threshold will stop the iterative smoothing procedure earlier, but may lead to less smooth midlines. Spline spacing and order might have to be tweaked depending on sample morphology. 
- Show Boundaries and Show Midlines can be ticked to generate napari shape layers with the relevant parameters.

There are fewer parameters in the Measure360 tab:

- Number of Lines: basically, how many sectors to split the 360 degrees around the centroid in.
- Filter Results: choose if you want the software to apply smoothing or not
- Filter Options:
    - Derivative
    - Stdev
- Threshold: Measurements are discarded if difference (if derivative) or stdev from previous point are above this value.


<div style="text-align: center;">
    <video controls autoplay muted width="640" height="360" preload="metadata">
   <source src="../run-analysis.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>
</div>

## Interacting with your results
You can interact with your results in several ways, here are some examples:

### Interactive filtering
You can apply interactive filtering by moving the sliders in the filter tab, cells which have a red box around them are discarded from the analasis and won't appear in the results table.

For now only a limited set of filters are available, but we aim to make this more customisable in the future.

<div style="text-align: center;">
    <video controls autoplay muted width="640" height="360" preload="metadata">
    <source src="../interactive-filtering.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>
</div>

### Highlighting selected cells
If you are interested in highlighting a specific cell with an odd value in the results table, you can do so by activating the option `Show Selected` in the mask layer, and then clicking on the row of interest. Only the mask of the cell linked to that row will be shown

<div style="text-align: center;">
    <video controls autoplay muted width="640" height="360" preload="metadata">
    <source src="../interactive-filtering.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>
</div>

### Histogram Plotting
We have included basic plotting functionalities in micromorph. You can select a column and then click "Plot Histogram..." to generate a histogram of the data. This is currently not customisable but we aim to make it so in the future.

<div style="text-align: center;">
    <video controls autoplay muted width="640" height="360" preload="metadata">
    <source src="../plotting-histograms.mp4" type="video/mp4">
    Your browser does not support the video tag.
    </video>
</div>

## Saving your results
You can save your results by clicking the dedicated buttons under the results table.

## Batch processing
Once you're happy with your analysis settings, you can choose to batch analyse multiple files and folders at once. If you are interested in doing this you can refer to the [batch tools guide](../usage/batch_tools_example.md). You can access the batch tools by running `micromorph-batch` or clicking on the "Batch Tools" button in the napari plugin, which opens the same GUI.