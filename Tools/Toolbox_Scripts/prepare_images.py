"""
This is the script for the Prepare Images tool, which is part of the Roof Damage Assessment
toolbox for Esri ArcGIS Pro. The script can be run using the Prepare Images tool or it can be
imported as a module and run by calling the main() function.

For full methodological details, please refer to the publication:

Kucharczyk, M., Nesbit, P. R., & Hugenholtz, C. H. (2025). Automated Mapping of Post-Storm
Roof Damage Using Deep Learning and Aerial Imagery: A Case Study in the Caribbean. Remote Sensing,
17(20), 3456. https://doi.org/10.3390/rs17203456

For usage instructions, please visit: https://github.com/maja-kucharczyk/roof-damage-assessment

Created by: Maja Kucharczyk

Version: 1.0.0 (2026-01-25)

This work is licensed under CC BY 4.0 (Creative Commons Attribution 4.0 International), a permissive
license allowing anyone to freely share, copy, adapt, and use material for any purpose, even
commercially, as long as they give proper attribution (credit) to the original creators, indicate
if changes were made, and provide a link to the license.
"""


# Imports
import os
import arcpy


def get_workspace_extension(
        workspace_path: str,
        ) -> str:
    """
    Returns the extension of a workspace path.
    
    Args:
        workspace_path (string): The path to the workspace.
        
    Returns:
        string: The extension of the workspace.
    """
    name_with_extension = os.path.basename(workspace_path)
    extension = os.path.splitext(name_with_extension)[1]
    return extension


def get_file_name(
        file_path: str,
        ) -> str:
    """
    Returns the name of a file without its extension.
    
    Args:
        file_path (string): The path to the file.
        
    Returns:
        string: The name of the file without its extension.
    """
    name_with_extension = os.path.basename(file_path)
    name_without_extension = os.path.splitext(name_with_extension)[0]
    return name_without_extension


def get_spatial_ref_type(
        data_path: str,
        ) -> str:
    """
    Returns the spatial reference type of a data element.
    
    Args:
        data_path (string): The path to the data element.
        
    Returns:
        string: The spatial reference type of the data element.
    """
    spatial_ref_type = arcpy.Describe(data_path).spatialreference.type
    return spatial_ref_type


def extract_rgb_bands(
        input_image_path: str,
        output_layer_path: str,
        ):
    """
    Extracts bands 1, 2, and 3 of an image and creates a temporary RGB raster layer.
    
    Args:
        input_image_path (string): The path to the input image.
        output_layer_path (string): The path to the output temporary RGB raster layer.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            pyramid = 'NONE',
            rasterStatistics = 'NONE',
            resamplingMethod = 'CUBIC'):
        arcpy.management.MakeRasterLayer(
            in_raster = input_image_path,
            out_rasterlayer = output_layer_path,
            band_index = '1;2;3',
            )


def get_spatial_ref_string(
        data_path: str,
        ) -> str:
    """
    Returns the spatial reference string of a data element.
    
    Args:
        data_path (string): The path to the data element.
        
    Returns:
        string: The spatial reference string of the data element.
    """
    spatial_ref_string = arcpy.Describe(data_path).spatialreference.exportToString()
    return spatial_ref_string


def project_resample_image(
        input_image_path: str,
        output_image_path: str,
        spatial_ref_string: str,
        ):
    """
    Projects an image and resamples it to 0.05 m/px using cubic convolution.
    
    Args:
        input_image_path (string): The path to the input image.
        output_image_path (string): The path to the output image.
        spatial_ref_string (string): The string of the output spatial reference.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            pyramid = 'NONE',
            rasterStatistics = 'NONE',
            resamplingMethod = 'CUBIC'):
        arcpy.management.ProjectRaster(
            in_raster = input_image_path,
            out_raster = output_image_path,
            out_coor_system = spatial_ref_string,
            resampling_type = 'CUBIC',
            cell_size = 0.05,
            )


def resample_image(
        input_image_path: str,
        output_image_path: str,
        ):
    """
    Resamples an image to 0.05 m/px using cubic convolution.
    
    Args:
        input_image_path (string): The path to the input image.
        output_image_path (string): The path to the output image.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            pyramid = 'NONE',
            rasterStatistics = 'NONE',
            resamplingMethod = 'CUBIC'):
        arcpy.management.Resample(
            in_raster = input_image_path,
            out_raster = output_image_path,
            cell_size = 0.05,
            resampling_type = 'CUBIC',
            )


def clip_image(
        input_image_path,
        output_image_path: str,
        clipping_fclass_path: str,
        ):
    """
    Clips an image using its boundary polygon.
    
    Args:
        input_image_path (string): The path to the input image.
        output_image_path (string): The path to the output image.
        clipping_fclass_path (string): The path to the feature class used for clipping the image.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            pyramid = 'NONE',
            rasterStatistics = 'NONE',
            resamplingMethod = 'CUBIC'):
        arcpy.management.Clip(
            in_raster = input_image_path,
            out_raster = output_image_path,
            in_template_dataset = clipping_fclass_path,
            clipping_geometry = 'ClippingGeometry',
            )


def export_image(
        input_image_path: str,
        output_image_path: str,
        ):
    """
    Exports an image to an 8-bit unsigned file geodatabase raster.
    
    Args:
        input_image_path (string): The path to the input image.
        output_image_path (string): The path to the output image.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            pyramid = 'PYRAMIDS -1 CUBIC LZ77 NO_SKIP',
            rasterStatistics = 'STATISTICS',
            resamplingMethod = 'CUBIC'):
        arcpy.management.CopyRaster(
            in_raster = input_image_path,
            out_rasterdataset = output_image_path,
            pixel_type = '8_BIT_UNSIGNED',
            )


def main(
        input_images_folder: str = arcpy.GetParameterAsText(0),
        boundary_polygons_gdb: str = arcpy.GetParameterAsText(1),
        output_images_gdb: str = arcpy.GetParameterAsText(2),
        scratch_gdb: str = arcpy.GetParameterAsText(3),
        ):
    """
    Prepares remote sensing images for deep learning model training and inference.
    
    The following processing is performed on each image: 
        • extracting RGB bands, if needed; 
        • projecting, if needed; 
        • resampling to 0.05 m/px using cubic convolution; 
        • clipping by the corresponding boundary polygon; and 
        • exporting to an 8-bit unsigned file geodatabase raster.
    
    Args:
        input_images_folder (string): The path to the input images folder.
        boundary_polygons_gdb (string): The path to the image boundary polygons file geodatabase.
        output_images_gdb (string): The path to the output file geodatabase for prepared images.
        scratch_gdb (string): The path to the scratch file geodatabase for intermediate outputs.
        
    Raises:
        FileNotFoundError: If a workspace does not exist or if an input image type is invalid.
        ValueError: If boundary_polygons_gdb, output_images_gdb, or scratch_gdb does not correspond
                    to a file geodatabase.
    """

    # If a workspace does not exist, end the process
    for workspace in [input_images_folder, boundary_polygons_gdb, output_images_gdb, scratch_gdb]:
        if os.path.exists(workspace):
            continue
        arcpy.AddError(f'{workspace} does not exist.')
        raise FileNotFoundError(f'{workspace} does not exist.')

    # If the boundary polygons path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = boundary_polygons_gdb) != '.gdb':
        arcpy.AddError('The boundary polygons path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The boundary polygons path must correspond to a file geodatabase (.gdb).')

    # If the output images path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = output_images_gdb) != '.gdb':
        arcpy.AddError('The output images path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The output images path must correspond to a file geodatabase (.gdb).')

    # If the scratch path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = scratch_gdb) != '.gdb':
        arcpy.AddError('The scratch path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The scratch path must correspond to a file geodatabase (.gdb).')

    # Set the scratch workspace to the scratch file geodatabase path
    arcpy.env.scratchWorkspace = scratch_gdb

    # Create a list of images to prepare
    arcpy.env.workspace = input_images_folder
    images = arcpy.ListRasters()

    # Count the total number of images
    images_remaining = len(images)

    # If there are zero valid images, end the process
    if images_remaining == 0:
        arcpy.AddError('The input images folder contains zero valid images. '
                       'Valid raster types include BMP, GIF, IMG, JP2, JPG, PNG, TIF, and GRID.')
        raise FileNotFoundError('The input images folder contains zero valid images. '
                       'Valid raster types include BMP, GIF, IMG, JP2, JPG, PNG, TIF, and GRID.')

    # Create a list of skipped images
    skipped_images = []

    # Create a list of image boundary polygon feature classes
    arcpy.env.workspace = boundary_polygons_gdb
    boundary_fclasses = arcpy.ListFeatureClasses()

    # Configure the tool's progress bar to increment by 33% since there are three
    # time-consuming preparation steps per image (resampling, clipping, exporting)
    arcpy.SetProgressor(
        type = 'step',
        message = '',
        min_range = 0,
        max_range = 100,
        step_value = 33,
        )

    for image in images:
        # Get the name of the image
        image_name = get_file_name(
            file_path = os.path.join(input_images_folder, image),
            )

        # If the image does not have a corresponding boundary feature class, skip the image
        if image_name not in boundary_fclasses:
            skipped_images.append(image)
            images_remaining -= 1
            arcpy.AddWarning(f'A feature class named {image_name} does not exist in the image '
                             f'boundary polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            print(f'A feature class named {image_name} does not exist in the image '
                             f'boundary polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            continue

        # Get the spatial reference type of the image
        image_sr_type = get_spatial_ref_type(
            data_path = os.path.join(input_images_folder, image),
            )

        # If the image's spatial reference is unknown, skip the image
        if image_sr_type == 'Unknown':
            skipped_images.append(image)
            images_remaining -= 1
            arcpy.AddWarning(f'The spatial reference of {image} is unknown and needs to be defined.'
                             f' The image has been skipped. {images_remaining} images remaining.')
            print(f'The spatial reference of {image} is unknown and needs to be defined.'
                             f' The image has been skipped. {images_remaining} images remaining.')
            continue

        # If the image's spatial reference is a geographic coordinate system, then the image is from
        # the NOAA NGS Emergency Response Imagery dataset and needs the following initial prep:
        if image_sr_type == 'Geographic':
            extract_rgb_bands(
                input_image_path = os.path.join(input_images_folder, image),
                output_layer_path = os.path.join(scratch_gdb, 'temp_raster_layer'),
                )
            boundary_sr_string = get_spatial_ref_string(
                data_path = os.path.join(boundary_polygons_gdb, image_name),
                )
            arcpy.SetProgressorLabel(f'Projecting and resampling {image}')
            print(f'Projecting and resampling {image}')
            project_resample_image(
                input_image_path = os.path.join(scratch_gdb, 'temp_raster_layer'),
                output_image_path = os.path.join(scratch_gdb, 'resampled'),
                spatial_ref_string = boundary_sr_string,
                )
            arcpy.SetProgressorPosition()

        # If the image's spatial reference is a projected coordinate system, then the image is from
        # the drone imagery dataset and needs the following initial prep:
        if image_sr_type == 'Projected':
            arcpy.SetProgressorLabel(f'Resampling {image}')
            print(f'Projecting and resampling {image}')
            resample_image(
                input_image_path = os.path.join(input_images_folder, image),
                output_image_path = os.path.join(scratch_gdb, 'resampled'),
                )
            arcpy.SetProgressorPosition()

        # All images need the following remaining prep:
        arcpy.SetProgressorLabel(f'Clipping {image}')
        print(f'Clipping {image}')
        clip_image(
            input_image_path = os.path.join(scratch_gdb, 'resampled'),
            output_image_path = os.path.join(scratch_gdb, 'clipped'),
            clipping_fclass_path = os.path.join(boundary_polygons_gdb, image_name),
            )
        arcpy.SetProgressorPosition()
        arcpy.SetProgressorLabel(f'Exporting {image}')
        print(f'Exporting {image}')
        export_image(
            input_image_path = os.path.join(scratch_gdb, 'clipped'),
            output_image_path = os.path.join(output_images_gdb, image_name),
            )

        # Update the progress indicators
        arcpy.SetProgressorPosition()
        images_remaining -= 1
        arcpy.AddMessage(f'Prepared {image}. {images_remaining} images remaining.')
        print(f'Prepared {image}. {images_remaining} images remaining.')
        arcpy.SetProgressorPosition(position = 0)

    # Confirm processing is done
    arcpy.AddMessage(f'\nExported prepared images to {output_images_gdb}')
    print(f'\nExported prepared images to {output_images_gdb}')

    # Report which images, if any, were skipped
    if len(skipped_images) > 0:
        arcpy.AddMessage('\nThe following images were skipped:')
        print('\nThe following images were skipped:')
        for image in skipped_images:
            arcpy.AddMessage(image)
            print(image)


# Prepare each image
if __name__ == '__main__':
    main()
