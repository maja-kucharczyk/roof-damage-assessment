"""
This is the script for the Export Training Data tool, which is part of the Roof Damage Assessment
toolbox for Esri ArcGIS Pro. The script can be run using the Export Training Data tool or it can be
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
import arcpy.ia


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


def export_training_data(
        input_image_path: str,
        output_folder_path: str,
        training_polygons_fclass_path: str,
        image_boundary_fclass_path: str,
        ):
    """
    Exports training data to the proper Esri format for training a Mask2Former model.
    
    Args:
        input_image_path (string): The path to the input image.
        output_folder_path (string): The path to the folder where the training data will be saved.
        training_polygons_fclass_path (string): The path to the training polygons feature class.
        image_boundary_fclass_path (string): The path to the image boundary feature class.
    """
    with arcpy.EnvManager(
            parallelProcessingFactor = '0',
            ):
        arcpy.ia.ExportTrainingDataForDeepLearning(
            in_raster = input_image_path,
            out_folder = output_folder_path,
            in_class_data = training_polygons_fclass_path,
            image_chip_format = 'TIFF',
            tile_size_x = 512,
            tile_size_y = 512,
            stride_x = 128,
            stride_y = 128,
            output_nofeature_tiles = False,
            metadata_format = 'Classified_Tiles',
            buffer_radius = 0,
            in_mask_polygons = image_boundary_fclass_path,
            rotation_angle = 0,
            reference_system = 'MAP_SPACE',
            processing_mode = 'PROCESS_AS_MOSAICKED_IMAGE',
            min_polygon_overlap_ratio = 0.5,
            )


def main(
        input_images_gdb: str = arcpy.GetParameterAsText(0),
        training_polygons_gdb: str = arcpy.GetParameterAsText(1),
        boundary_polygons_gdb: str = arcpy.GetParameterAsText(2),
        output_data_folder: str = arcpy.GetParameterAsText(3),
        ):
    """
    Exports training data to the proper Esri format for training a Mask2Former model.
    
    Args:
        input_images_gdb (string): The path to the prepared training images file geodatabase.
        training_polygons_gdb (string): The path to the training polygons file geodatabase.
        boundary_polygons_gdb (string): The path to the image boundary polygons file geodatabase.
        output_data_folder (string): The path to the output folder for exported training data.
        
    Raises:
        FileNotFoundError: If a workspace does not exist or if there are zero input images.
        ValueError: If input_images_gdb, boundary_polygons_gdb, or training_polygons_gdb does not
                    correspond to a file geodatabase.
    """

    # If a workspace does not exist, end the process
    for workspace in [input_images_gdb, boundary_polygons_gdb, training_polygons_gdb,
                      output_data_folder]:
        if os.path.exists(workspace):
            continue
        arcpy.AddError(f'{workspace} does not exist.')
        raise FileNotFoundError(f'{workspace} does not exist.')

    # If the input images path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = input_images_gdb) != '.gdb':
        arcpy.AddError('The input images path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The input images path must correspond to a file geodatabase (.gdb).')

    # If the training polygons path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = training_polygons_gdb) != '.gdb':
        arcpy.AddError('The training polygons path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The training polygons path must correspond to a file geodatabase (.gdb).')

    # If the boundary polygons path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = boundary_polygons_gdb) != '.gdb':
        arcpy.AddError('The boundary polygons path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The boundary polygons path must correspond to a file geodatabase (.gdb).')

    # Create a list of training images
    arcpy.env.workspace = input_images_gdb
    images = arcpy.ListRasters()

    # Count the total number of images
    images_remaining = len(images)

    # If there are zero images, end the process
    if images_remaining == 0:
        arcpy.AddError('The input images file geodatabase contains zero rasters.')
        raise FileNotFoundError('The input images file geodatabase contains zero rasters.')

    # Create a list of skipped images
    skipped_images = []

    # Create a list of training polygon feature classes
    arcpy.env.workspace = training_polygons_gdb
    training_polygons_fclasses = arcpy.ListFeatureClasses()

    # Create a list of image boundary polygon feature classes
    arcpy.env.workspace = boundary_polygons_gdb
    boundary_fclasses = arcpy.ListFeatureClasses()

    # Configure the tool's progress bar to increment by 100% divided by the total number of images
    arcpy.SetProgressor(
        type = 'step',
        message = '',
        min_range = 0,
        max_range = 100,
        step_value = int(100/images_remaining),
        )

    # For each image:
    for image in images:

        # If the image does not have a corresponding training polygons feature class, skip the image
        if image not in training_polygons_fclasses:
            arcpy.SetProgressorPosition()
            skipped_images.append(image)
            images_remaining -= 1
            arcpy.AddWarning(f'A feature class named {image} does not exist in the training '
                             f'polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            print(f'A feature class named {image} does not exist in the training '
                             f'polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            continue

        # If the image does not have a corresponding boundary feature class, skip the image
        if image not in boundary_fclasses:
            skipped_images.append(image)
            arcpy.SetProgressorPosition()
            images_remaining -= 1
            arcpy.AddWarning(f'A feature class named {image} does not exist in the image '
                             f'boundary polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            print(f'A feature class named {image} does not exist in the image '
                             f'boundary polygons file geodatabase. {image} has been skipped. '
                             f'{images_remaining} images remaining.')
            continue

        # If the image passes all checks, export training data from the image
        arcpy.SetProgressorLabel(f'Exporting training data from {image}')
        print(f'Exporting training data from {image}')
        export_training_data(
                input_image_path = os.path.join(input_images_gdb, image),
                output_folder_path = output_data_folder,
                training_polygons_fclass_path = os.path.join(training_polygons_gdb, image),
                image_boundary_fclass_path = os.path.join(boundary_polygons_gdb, image),
                )

        # Update the progress indicators
        arcpy.SetProgressorPosition()
        images_remaining -= 1
        arcpy.AddMessage(f'Exported training data from {image}. '
                         f'{images_remaining} images remaining.')
        print(f'Exported training data from {image}. '
                         f'{images_remaining} images remaining.')

    # Confirm processing is done
    arcpy.AddMessage(f'\nExported training data to {output_data_folder}')
    print(f'\nExported training data to {output_data_folder}')

    # Report which images, if any, were skipped
    if len(skipped_images) > 0:
        arcpy.AddMessage('\nThe following images were skipped:')
        print('\nThe following images were skipped:')
        for image in skipped_images:
            arcpy.AddMessage(image)
            print(image)


# Export training data from each image
if __name__ == '__main__':
    main()
