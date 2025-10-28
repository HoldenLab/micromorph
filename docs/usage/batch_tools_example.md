# Batch processing the data
!!! warning
    This page is still work in progress, so some parts may be incomplete or missing. We are working on it!

A batch processing panel is available if `micromorph` was installed on your environment with ```pip install micromorph[gui]```.

After that, the panel can opened by running the command

`micromorph-batch`

<figure markdown="1">
![batchtools](../resources/batch_tools_panel.png)
</figure>

You can add folders to be analysed by copying their path into the text box and clicking `Add Folder`.
The batch utility will specifically look for `.tif` files in the selected folders and subfolders. We are looking at ways to make this suitable for a broader format of files.
If `Run Segmentation` is checked, the segmentation will be run on the selected folders and save masks in the same 
folder as the images, with the same name as the image but with `_mask` appended.

If `Load Segmentation` is checked, the batch utility will look for masks in the selected folders, assuming the same 
naming convention described above.

`Analysis Settings` and `Segmentation Settings` are the same as the settings window in the main napari plugin (see 
[here](./napari_plugin_example.md) for more information)., 
while 
`Image Loading Settings` 
allows some extra flexibility with regards to how images are loaded. In general you shouldn't need to change it but 
may want to if, for example, you only want to analyse a specific half of the image etc.