"""
This is the script for the Delineate Roof Damage tool, which is part of the Roof Damage Assessment
toolbox for Esri ArcGIS Pro. The script can be run using the Delineate Roof Damage tool or it can be
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


def generate_classified_raster(
        input_image_path: str,
        input_model_path: str,
        ):
    """
    Generates a classified raster using an input image and trained deep learning model.
    
    Args:
        input_image_path (string): The path to the input image.
        input_model_path (string): The path to the trained deep learning model (.emd or .dlpk).
        
    Returns:
        classified_raster: A raster with cell values corresponding to the class(es) of the model.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            processorType = 'GPU',
            gpuId = 0,
            ):
        classified_raster = arcpy.ia.ClassifyPixelsUsingDeepLearning(
            in_raster = input_image_path,
            in_model_definition = input_model_path,
            arguments =
            'batch_size 4; padding 128; predict_background False; test_time_augmentation False',
            )
    return classified_raster


def raster_to_fclass(
        input_raster_path: str,
        output_fclass_path: str,
        ):
    """
    Converts a classified raster to a feature class.
    Contiguous cells of the same class are grouped to form single-part polygons.
    
    Args:
        input_raster_path (string): The path to the input raster.
        output_fclass_path (string): The path to the output feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.conversion.RasterToPolygon(
            in_raster = input_raster_path,
            out_polygon_features = output_fclass_path,
            simplify = 'NO_SIMPLIFY',
            raster_field = 'Class',
            create_multipart_features = 'SINGLE_OUTER_PART')


def delete_fclass_fields(
        input_fclass_path: str,
        ):
    """
    Deletes the "Id" and "gridcode" fields from a feature class.
    
    Args:
        input_fclass_path (string): The path to the input feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.DeleteField(
            in_table = input_fclass_path,
            drop_field = ['Id', 'gridcode'],
            method = 'DELETE_FIELDS')


def merge_fclasses(
        fclass_paths_list: list,
        output_fclass_path: str,
        ):
    """
    Merges feature classes into one feature class.
    
    Args:
        fclass_paths_list (list): A list of (string) paths to feature classes.
        output_fclass_path (string): The path to the output feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.Merge(
            inputs = fclass_paths_list,
            output = output_fclass_path,
            add_source = 'NO_SOURCE_INFO',
            field_match_mode = 'AUTOMATIC',
            )


def main(
        input_images_gdb: str = arcpy.GetParameterAsText(0),
        model_path_decking: str = arcpy.GetParameterAsText(1),
        model_path_hole: str = arcpy.GetParameterAsText(2),
        model_path_dual: str = arcpy.GetParameterAsText(3),
        output_fclasses_gdb: str = arcpy.GetParameterAsText(4),
        scratch_gdb: str = arcpy.GetParameterAsText(5),
        ):
    """
    Exports training data to the proper Esri format for training a Mask2Former model.
    
    Args:
        input_images_gdb (string): The path to the prepared test images file geodatabase.
        model_path_decking (string): The path to the trained single-class roof decking model 
        (.emd or .dlpk).
        model_path_hole (string): The path to the trained single-class roof hole model 
        (.emd or .dlpk).
        model_path_dual (string): The path to the trained dual-class (decking and hole) model 
        (.emd or .dlpk).
        output_fclasses_gdb (string): The path to the output file geodatabase for predicted 
        polygons.
        scratch_gdb (string): The path to the scratch file geodatabase for intermediate outputs.
        
    Raises:
        FileNotFoundError: If a workspace/model does not exist or if there are zero input images.
        ValueError: If input_images_gdb, output_fclasses_gdb, or scratch_gdb does not correspond 
                    to a file geodatabase.
    """

    # If a workspace does not exist, end the process
    for path in [input_images_gdb, output_fclasses_gdb, scratch_gdb]:
        if os.path.exists(path):
            continue
        arcpy.AddError(f'{path} does not exist.')
        raise FileNotFoundError(f'{path} does not exist.')

    # If there are zero input model paths, end the process
    model_paths = []
    for path in [model_path_decking, model_path_hole, model_path_dual]:
        if path == '':
            continue
        model_paths.append(path)
    if len(model_paths) == 0:
        arcpy.AddError('There are zero input models. Input at least one model path '
                       '(.emd or .dlpk).')
        raise FileNotFoundError('There are zero input models. Input at least one model path '
                       '(.emd or .dlpk).')

    # If a model path does not exist, end the process
    for path in model_paths:
        if os.path.exists(path):
            continue
        arcpy.AddError(f'{path} does not exist.')
        raise FileNotFoundError(f'{path} does not exist.')

    # If the input images path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = input_images_gdb) != '.gdb':
        arcpy.AddError('The input images path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The input images path must correspond to a file geodatabase (.gdb).')

    # If the output feature classes path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = output_fclasses_gdb) != '.gdb':
        arcpy.AddError('The output path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The output path must correspond to a file geodatabase (.gdb).')

    # If the scratch path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = scratch_gdb) != '.gdb':
        arcpy.AddError('The scratch path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The scratch path must correspond to a file geodatabase (.gdb).')

    # Set the scratch workspace to the scratch file geodatabase path
    arcpy.env.scratchWorkspace = scratch_gdb

    # Create a list of test images
    arcpy.env.workspace = input_images_gdb
    images = arcpy.ListRasters()

    # Count the total number of images
    images_remaining = len(images)

    # If there are zero images, end the process
    if images_remaining == 0:
        arcpy.AddError('The input images file geodatabase contains zero rasters.')
        raise FileNotFoundError('The input images file geodatabase contains zero rasters.')

    # create dictionary of input models
    model_dictionary = {}

    if model_path_decking != '':
        model_dictionary['decking'] = model_path_decking

    if model_path_hole != '':
        model_dictionary['hole'] = model_path_hole

    if model_path_dual != '':
        model_dictionary['dual'] = model_path_dual

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

        arcpy.SetProgressorLabel(f'Delineating roof damage in {image}')
        print(f'Delineating roof damage in {image}')

        # Create a list to store the path to the roof damage feature class predicted by each model
        predicted_fclass_paths = []

        # For each model:
        for damage_class, model_path in model_dictionary.items():

            # Generate a roof damage raster
            roof_damage_raster = generate_classified_raster(
                input_image_path = os.path.join(input_images_gdb, image),
                input_model_path = model_path,
                )

            # Convert the roof damage raster to a single-part polygon feature class
            raster_to_fclass(
                input_raster_path = roof_damage_raster,
                output_fclass_path = os.path.join(scratch_gdb, f'{image}_{damage_class}'),
                )

            # Delete the "Id" and "gridcode" fields from the polygon feature class
            delete_fclass_fields(
                input_fclass_path = os.path.join(scratch_gdb, f'{image}_{damage_class}'),
                )

            # Append the feature class to the predicted feature class paths list
            predicted_fclass_paths.append(os.path.join(scratch_gdb, f'{image}_{damage_class}'))

        # Merge the model-specific feature classes into one feature class per image
        merge_fclasses(
            fclass_paths_list = predicted_fclass_paths,
            output_fclass_path = os.path.join(output_fclasses_gdb, image),
            )

        # Update the progress indicators
        arcpy.SetProgressorPosition()
        images_remaining -= 1
        arcpy.AddMessage(f'Delineated roof damage in {image}. {images_remaining} images remaining.')
        print(f'Delineated roof damage in {image}. {images_remaining} images remaining.')

    # Confirm processing is done
    arcpy.AddMessage(f'\nExported predicted polygons feature classes to {output_fclasses_gdb}')
    print(f'\nExported predicted polygons feature classes to {output_fclasses_gdb}')


# Export training data from each image
if __name__ == '__main__':
    main()
