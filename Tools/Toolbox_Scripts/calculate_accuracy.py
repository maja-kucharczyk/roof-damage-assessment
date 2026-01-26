"""
This is the script for the Calculate Accuracy tool, which is part of the Roof Damage Assessment
toolbox for Esri ArcGIS Pro. The script can be run using the Calculate Accuracy tool or it can be
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


def dissolve_fclass_by_class(
        input_fclass_path: str,
        output_fclass_path: str,
        class_field_name: str,
        ):
    """
    Dissolves a feature class based on a class field.
    
    Args:
        input_fclass_path (string): The path to the input feature class.
        output_fclass_path (string): The path to the output feature class.
        class_field_name (string): The name of the class field.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.Dissolve(
            in_features = input_fclass_path,
            out_feature_class = output_fclass_path,
            dissolve_field = class_field_name,
            multi_part = 'MULTI_PART',
            )


def create_layer_by_class(
        input_fclass_path: str,
        output_layer_name: str,
        class_field_name: str,
        class_name: str,
        ):
    """
    Converts one class of an input feature class to a feature layer.
    
    Args:
        input_fclass_path (string): The path to the input feature class.
        output_layer_name (string): The name of the output feature layer.
        class_field_name (string): The name of the class field in the input feature class.
        class_name (string): The name of the class from which to create a feature layer.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.MakeFeatureLayer(
            in_features = input_fclass_path,
            out_layer = output_layer_name,
            where_clause = f"{class_field_name} = '{class_name}'",
            )


def layer_to_fclass(
        input_layer_name: str,
        output_fclass_path: str,
        ):
    """
    Converts a feature layer to a feature class.
    
    Args:
        input_layer_name (string): The name of the input feature layer.
        output_fclass_path (string): The path to the output feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CopyFeatures(
            in_features = input_layer_name,
            out_feature_class = output_fclass_path,
            )


def layer_to_raster(
        input_layer_name: str,
        class_field_name: str,
        snap_raster_path: str,
        output_raster_path: str,
        ):
    """
    Converts a feature layer to a classified raster using the cell boundaries of a snap raster.
    
    Args:
        input_layer_name (string): The name of the input feature layer.
        class_field_name (string): The name of the class field in the input feature layer.
        snap_raster_path (string): The path to the snap raster.
        output_raster_path (string): The path to the output classified raster.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            snapRaster = snap_raster_path,
            ):
        arcpy.conversion.PolygonToRaster(
            in_features = input_layer_name,
            value_field = class_field_name,
            out_rasterdataset = output_raster_path,
            cell_assignment = 'CELL_CENTER',
            cellsize = snap_raster_path,
            build_rat = 'DO_NOT_BUILD',
            )


def raster_to_fclass(
        input_raster_path: str,
        class_field_name: str,
        output_fclass_path: str,
        ):
    """
    Converts a raster to a multipart polygon feature class.
    
    Args:
        input_raster_path (string): The path to the input raster.
        class_field_name (string): The name of the field in the input raster that is used 
        for assigning classes to the polygons in the output feature class. Polygons of
        the same class will be dissolved.
        output_fclass_path (string): The path to the output feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.conversion.RasterToPolygon(
            in_raster = input_raster_path,
            out_polygon_features = output_fclass_path,
            simplify = 'NO_SIMPLIFY',
            raster_field = class_field_name,
            create_multipart_features = 'MULTIPLE_OUTER_PART',
            )


def create_union_fclass(
        predicted_fclass_path: str,
        reference_fclass_path: str,
        output_fclass_path: str,
        ):
    """
    Creates a union feature class from a predicted polygons feature class and 
    a reference polygons feature class.
    
    Args:
        predicted_fclass_path (string): The path to the predicted polygons feature class.
        reference_fclass_path (string): The path to the reference polygons feature class.
        output_fclass_path (string): The path to the output union feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.analysis.Union(
            in_features = [predicted_fclass_path, reference_fclass_path],
            out_feature_class = output_fclass_path,
            join_attributes = 'ONLY_FID',
            gaps = 'GAPS',
            )


def calculate_accuracy_category_field(
        input_fclass_path: str,
        predicted_fclass_name: str,
        reference_fclass_name: str,
        ):
    """
    Creates an "Accuracy_Category" field in an input union feature class and assigns each
    feature a value of "TP" (true positive), "FP" (false positive), or "FN" (false negative).
    
    Args:
        input_fclass_path (string): The path to the input union feature class.
        predicted_fclass_name (string): The name of the predicted polygons feature class that was
        used to create the union feature class.
        reference_fclass_name (string): The name of the reference polygons feature class that was
        used to create the union feature class.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_fclass_path,
            field = 'Accuracy_Category',
            expression =
            f'accuracy_category(!FID_{predicted_fclass_name}!,!FID_{reference_fclass_name}!)',
            expression_type = 'PYTHON3',
            code_block =
            f'''def accuracy_category(FID_{predicted_fclass_name}, FID_{reference_fclass_name}):
                if FID_{predicted_fclass_name} == 1 and FID_{reference_fclass_name} == 1:
                    return "TP"
                elif FID_{predicted_fclass_name} == 1 and FID_{reference_fclass_name} == -1:
                    return "FP"
                elif FID_{predicted_fclass_name} == -1 and FID_{reference_fclass_name} == 1:
                    return "FN"
                else:
                    return "None"''',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def create_pixels_per_category_table(
        input_fclass_path: str,
        input_image_path: str,
        output_table_path: str,
        ):
    """
    Counts the number of image pixels inside each feature (i.e., accuracy category)
    of an input union feature class and outputs a table with the data.
    
    Args:
        input_fclass_path (string): The path to the input union feature class.
        input_image_path (string): The path to the image used for pixel counting.
        output_table_path (string): The path to the output table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.ia.ZonalStatisticsAsTable(
            in_zone_data = input_fclass_path,
            zone_field = 'Accuracy_Category',
            in_value_raster = input_image_path,
            out_table = output_table_path,
            ignore_nodata = 'DATA',
            statistics_type = 'MAJORITY_VALUE_COUNT_PERCENT',
            process_as_multidimensional = 'CURRENT_SLICE',
            )


def calculate_zone_code_field(
        input_table_path: str,
        ):
    """
    Assigns a constant value to the "ZONE_CODE" field of an input table.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'ZONE_CODE',
            expression = 1,
            expression_type = 'PYTHON3',
            field_type = 'SHORT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def pivot_table(
        input_table_path: str,
        output_table_path: str,
        ):
    """
    Pivots an input table using the constant "ZONE_CODE" field as the input field (such that all
    records in the input table form one record in the output table), the "Accuracy_Category" field
    as the pivot field (resulting in one field per accuracy category in the output table), and the
    "COUNT" field as the value field (to populate each accuracy category field with its pixel count).
    
    Args:
        input_table_path (string): The path to the input table.
        output_table_path (string): The path to the output table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.PivotTable(
            in_table = input_table_path,
            fields = 'ZONE_CODE',
            pivot_field = 'Accuracy_Category',
            value_field = 'COUNT',
            out_table = output_table_path,
            )


def delete_zone_code_field(
        input_table_path: str,
        ):
    """
    Deletes the "ZONE_CODE" field from an input table.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.DeleteField(
            in_table = input_table_path,
            drop_field = 'ZONE_CODE',
            method = 'DELETE_FIELDS',
            )


def calculate_image_field(
        input_table_path: str,
        image_name: str,
        ):
    """
    Creates an "Image" field in an input table and assigns it the input image name.
    
    Args:
        input_table_path (string): The path to the input table.
        image_name (string): The image name to assign to the "Image" field.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'Image',
            expression = f'"{image_name}"',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def calculate_class_field(
        input_table_path: str,
        class_name: str,
        ):
    """
    Creates a "Class" field in an input table and assigns it the input damage class name.
    
    Args:
        input_table_path (string): The path to the input table.
        class_name (string): The damage class name to assign to the "Class" field.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'Class',
            expression = f'"{class_name}"',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def verify_accuracy_category_fields(
        input_table_path: str,
        ):
    """
    Evaluates whether an input table has "TP", "FP", and "FN" fields. If a field is missing,
    it is created and assigned a value of 0.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    table_field_names = [field.name for field in arcpy.ListFields(input_table_path)]
    for accuracy_category in ['TP', 'FP', 'FN']:
        if accuracy_category not in table_field_names:
            with arcpy.EnvManager(
                    overwriteOutput = True,
                    ):
                arcpy.management.CalculateField(
                    in_table = input_table_path,
                    field = accuracy_category,
                    expression = 0,
                    expression_type = 'PYTHON3',
                    field_type = 'DOUBLE',
                    enforce_domains = 'NO_ENFORCE_DOMAINS',
                    )


def merge_accuracy_tables(
        input_table_paths: list,
        output_table_path: str,
        ):
    """
    Merges image-specific accuracy tables into one table.
    
    Args:
        input_table_paths (list): A list of paths to the input tables.
        output_table_path (string): The path to the output table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.Merge(
            inputs = input_table_paths,
            output = output_table_path,
            add_source = 'NO_SOURCE_INFO',
            field_match_mode = 'AUTOMATIC',
            )


def add_summary_row(
        input_table_path: str,
        class_name: str,
        ):
    """
    Inserts a summary row in an input table, assigns the sum of "TP" to the "TP" field,
    assigns the sum of "FP" to the "FP" field, assigns the sum of "FN" to the "FN" field,
    assigns a value of "All" to the "Image" field, and assigns the input damage class name
    to the "Class" field.
    
    Args:
        input_table_path (string): The path to the input table.
        class_name (string): The damage class name to assign to the "Class" field.
    """
    sum_tp = sum(row[0] for row in arcpy.da.SearchCursor(input_table_path, 'TP'))
    sum_fp = sum(row[0] for row in arcpy.da.SearchCursor(input_table_path, 'FP'))
    sum_fn = sum(row[0] for row in arcpy.da.SearchCursor(input_table_path, 'FN'))
    fields = ['TP', 'FP', 'FN', 'Image', 'Class']
    with arcpy.da.InsertCursor(input_table_path, fields) as cursor:
        cursor.insertRow([sum_tp, sum_fp, sum_fn, 'All', class_name])


def calculate_union_field(
        input_table_path: str,
        ):
    """
    Creates a "Union" field in an input table and calculates it using the "TP", "FP", and
    "FN" fields.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'Union',
            expression = '!TP! + !FP! + !FN!',
            expression_type = 'PYTHON3',
            field_type = 'LONG',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def calculate_precision_field(
        input_table_path: str,
        ):
    """
    Creates a "Precision" field in an input table and calculates it using the "TP" and "FP" fields.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'Precision',
            expression = '"%.2f" % (!TP! / (!TP! + !FP!))',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def calculate_recall_field(
        input_table_path: str,
        ):
    """
    Creates a "Recall" field in an input table and calculates it using the "TP" and "FN" fields.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'Recall',
            expression = '"%.2f" % (!TP! / (!TP! + !FN!))',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def calculate_f1_field(
        input_table_path: str,
        ):
    """
    Creates an "F1" field in an input table and calculates it using the "TP", "FP", and "FN" fields.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'F1',
            expression = '"%.2f" % ((2 * !TP!) / ((2 * !TP!) + !FP! + !FN!))',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def calculate_iou_field(
        input_table_path: str,
        ):
    """
    Creates an "IoU" field in an input table and calculates it using the "TP", "FP", and
    "FN" fields.
    
    Args:
        input_table_path (string): The path to the input table.
    """
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.management.CalculateField(
            in_table = input_table_path,
            field = 'IoU',
            expression = '"%.2f" % (!TP! / (!TP! + !FP! + !FN!))',
            expression_type = 'PYTHON3',
            field_type = 'TEXT',
            enforce_domains = 'NO_ENFORCE_DOMAINS',
            )


def improve_field_settings(
        input_table_path: str,
        output_table_path: str,
        ):
    """
    Converts an input accuracy table to a final accuracy table with improved field settings.
    
    Args:
        input_table_path (string): The path to the input table.
        output_table_path (string): The path to the output table.
    """
    fieldmapping = arcpy.FieldMappings()
    fieldmapping_string = f'''
    Image "Image" false true false 255 Text -1 -1,First,#,{input_table_path},Image,-1,-1;
    Class "Class" false true false 255 Text -1 -1,First,#,{input_table_path},Class,-1,-1;
    TP "TP" false true false -1 Long -1 -1,First,#,{input_table_path},TP,-1,-1;
    FP "FP" false true false -1 Long -1 -1,First,#,{input_table_path},FP,-1,-1;
    FN "FN" false true false -1 Long -1 -1,First,#,{input_table_path},FN,-1,-1;
    Union "Union" false true false -1 Long -1 -1,First,#,{input_table_path},Union,-1,-1;
    Precision "Precision" false true false -1 Double 5 2,First,#,{input_table_path},Precision,-1,-1;
    Recall "Recall" false true false -1 Double 5 2,First,#,{input_table_path},Recall,-1,-1;
    F1 "F1" false true false -1 Double 5 2,First,#,{input_table_path},F1,-1,-1;
    IoU "IoU" false true false -1 Double 5 2,First,#,{input_table_path},IoU,-1,-1
    '''
    fieldmapping.loadFromString(fieldmapping_string)
    with arcpy.EnvManager(
            overwriteOutput = True,
            ):
        arcpy.conversion.TableToTable(
            in_rows = input_table_path,
            out_path = os.path.dirname(output_table_path),
            out_name = os.path.basename(output_table_path),
            field_mapping = fieldmapping,
            )


def main(
        predicted_polygons_gdb: str = arcpy.GetParameterAsText(0),
        reference_polygons_gdb: str = arcpy.GetParameterAsText(1),
        prepared_test_images_gdb: str = arcpy.GetParameterAsText(2),
        output_tables_gdb: str = arcpy.GetParameterAsText(3),
        scratch_gdb: str = arcpy.GetParameterAsText(4),
        ):
    """
    Exports training data to the proper Esri format for training a Mask2Former model.

    Args:
        predicted_polygons_gdb (string): The path to the predicted polygons file geodatabase.
        reference_polygons_gdb (string): The path to the reference polygons file geodatabase.
        prepared_test_images_gdb (string): The path to the prepared test images file geodatabase.
        output_tables_gdb (string): The path to the output file geodatabase for accuracy tables.
        scratch_gdb (string): The path to the scratch file geodatabase for intermediate outputs.

    Raises:
        FileNotFoundError: If a workspace does not exist or if there are zero predicted polygons
        feature classes.
        ValueError: If predicted_polygons_gdb, reference_polygons_gdb, prepared_test_images_gdb,
        output_tables_gdb, or scratch_gdb does not correspond to a file geodatabase.
    """

    # If a workspace does not exist, end the process
    for path in [predicted_polygons_gdb, reference_polygons_gdb, prepared_test_images_gdb,
                 output_tables_gdb, scratch_gdb]:
        if os.path.exists(path):
            continue
        arcpy.AddError(f'{path} does not exist.')
        raise FileNotFoundError(f'{path} does not exist.')

    # If the predicted polygons path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = predicted_polygons_gdb) != '.gdb':
        arcpy.AddError('The predicted polygons path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The predicted polygons path must correspond to a file geodatabase (.gdb).')

    # If the reference polygons path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = reference_polygons_gdb) != '.gdb':
        arcpy.AddError('The reference polygons path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The reference polygons path must correspond to a file geodatabase (.gdb).')

    # If the prepared test images path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = prepared_test_images_gdb) != '.gdb':
        arcpy.AddError('The prepared test images path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The prepared test images path must correspond to a file geodatabase (.gdb).')

    # If the output accuracy tables path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = output_tables_gdb) != '.gdb':
        arcpy.AddError('The output accuracy tables path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The output accuracy tables path must correspond to a file geodatabase (.gdb).')

    # If the scratch path is not a file geodatabase, end the process
    if get_workspace_extension(workspace_path = scratch_gdb) != '.gdb':
        arcpy.AddError('The scratch path must correspond to a file geodatabase (.gdb).')
        raise ValueError('The scratch path must correspond to a file geodatabase (.gdb).')

    # Set the scratch workspace to the scratch file geodatabase path
    arcpy.env.scratchWorkspace = scratch_gdb

    # Create a list of predicted polygons feature classes to evaluate
    arcpy.env.workspace = predicted_polygons_gdb
    predicted_polygons_fclasses = arcpy.ListFeatureClasses()

    # Count the total number of predicted polygons feature classes
    fclasses_remaining = len(predicted_polygons_fclasses)

    # If there are zero predicted polygons feature classes, end the process
    if fclasses_remaining == 0:
        arcpy.AddError('The predicted polygons file geodatabase contains zero feature classes.')
        raise FileNotFoundError('The predicted polygons file geodatabase contains zero feature classes.')

    # Create a list of skipped feature classes
    skipped_fclasses = []

    # Create a list of reference polygons feature classes
    arcpy.env.workspace = reference_polygons_gdb
    reference_polygons_fclasses = arcpy.ListFeatureClasses()

    # Create a list of prepared test images
    arcpy.env.workspace = prepared_test_images_gdb
    prepared_test_images = arcpy.ListRasters()

    # Create a list of damage classes
    damage_classes = ['Decking', 'Hole']

    # Configure the tool's progress bar to increment by 100% divided by the total number of
    # predicted polygons feature classes
    arcpy.SetProgressor(
        type = 'step',
        message = '',
        min_range = 0,
        max_range = 100,
        step_value = int(100/fclasses_remaining),
        )

    # Calculate the accuracy of each predicted polygons feature class:
    for fclass_name in predicted_polygons_fclasses:

        # If the predicted polygons feature class does not have a corresponding reference polygons
        # feature class, skip the feature class
        if fclass_name not in reference_polygons_fclasses:
            arcpy.SetProgressorPosition()
            skipped_fclasses.append(fclass_name)
            fclasses_remaining -= 1
            arcpy.AddWarning(f'A feature class named {fclass_name} does not exist in the reference '
                             f'polygons file geodatabase. {fclass_name} has been skipped. '
                             f'{fclasses_remaining} predicted polygons feature classes remaining.')
            print(f'A feature class named {fclass_name} does not exist in the reference '
                             f'polygons file geodatabase. {fclass_name} has been skipped. '
                             f'{fclasses_remaining} predicted polygons feature classes remaining.')
            continue

        # If the predicted polygons feature class does not have a corresponding prepared test
        # image, skip the feature class
        if fclass_name not in prepared_test_images:
            arcpy.SetProgressorPosition()
            skipped_fclasses.append(fclass_name)
            fclasses_remaining -= 1
            arcpy.AddWarning(f'An image named {fclass_name} does not exist in the prepared test '
                             f'images file geodatabase. {fclass_name} has been skipped. '
                             f'{fclasses_remaining} predicted polygons feature classes remaining.')
            print(f'An image named {fclass_name} does not exist in the prepared test '
                             f'images file geodatabase. {fclass_name} has been skipped. '
                             f'{fclasses_remaining} predicted polygons feature classes remaining.')
            continue

        # If the feature class passes all checks, evaluate the feature class
        arcpy.SetProgressorLabel(f'Calculating accuracy of {fclass_name}')
        print(f'Calculating accuracy of {fclass_name}')

        # Dissolve each predicted polygons feature class by damage class
        dissolve_fclass_by_class(
                input_fclass_path = os.path.join(predicted_polygons_gdb, fclass_name),
                output_fclass_path = os.path.join(scratch_gdb, 'predicted_polygons_dissolve'),
                class_field_name = 'Class',
                )

        # Dissolve each reference polygons feature class by damage class
        dissolve_fclass_by_class(
                input_fclass_path = os.path.join(reference_polygons_gdb, fclass_name),
                output_fclass_path = os.path.join(scratch_gdb, 'reference_polygons_dissolve'),
                class_field_name = 'ClassName',
                )

        # Calculate the accuracy of one damage class at a time
        for damage_class in damage_classes:
            # Create a temporary feature layer of dissolved predicted polygons
            create_layer_by_class(
                    input_fclass_path = os.path.join(scratch_gdb, 'predicted_polygons_dissolve'),
                    output_layer_name = f'predicted_polygons_dissolve_{damage_class}',
                    class_field_name = 'Class',
                    class_name = damage_class,
                    )

            # Export the temporary feature layer to a feature class
            layer_to_fclass(
                    input_layer_name = f'predicted_polygons_dissolve_{damage_class}',
                    output_fclass_path = os.path.join(scratch_gdb, f'predicted_{damage_class}'),
                    )

            # Create a temporary feature layer of dissolved reference polygons
            create_layer_by_class(
                    input_fclass_path = os.path.join(scratch_gdb, 'reference_polygons_dissolve'),
                    output_layer_name = f'reference_polygons_dissolve_{damage_class}',
                    class_field_name = 'ClassName',
                    class_name = damage_class,
                    )

            # Convert the temporary feature layer to a raster using the test image as a snap raster
            layer_to_raster(
                    input_layer_name = f'reference_polygons_dissolve_{damage_class}',
                    class_field_name = 'ClassName',
                    snap_raster_path = os.path.join(prepared_test_images_gdb, fclass_name),
                    output_raster_path = os.path.join(scratch_gdb, f'reference_raster_{damage_class}'),
                    )

            # Convert the raster to a feature class
            raster_to_fclass(
                    input_raster_path = os.path.join(scratch_gdb, f'reference_raster_{damage_class}'),
                    class_field_name = 'ClassName',
                    output_fclass_path = os.path.join(scratch_gdb, f'reference_{damage_class}'),
                    )

            # Create a union feature class using the predicted and reference feature classes
            create_union_fclass(
                    predicted_fclass_path = os.path.join(scratch_gdb, f'predicted_{damage_class}'),
                    reference_fclass_path = os.path.join(scratch_gdb, f'reference_{damage_class}'),
                    output_fclass_path = os.path.join(scratch_gdb, f'union_{damage_class}'),
                    )

            # Create an "Accuracy_Category" field and assign each union feature a label of:
            # "TP" (true positive) if it is the intersection of the predicted and reference polygons,
            # "FP" (false positive) if it is the predicted polygon only, or
            # "FN" (false negative) if it is the reference polygon only.
            calculate_accuracy_category_field(
                    input_fclass_path = os.path.join(scratch_gdb, f'union_{damage_class}'),
                    predicted_fclass_name = f'predicted_{damage_class}',
                    reference_fclass_name = f'reference_{damage_class}',
                    )

            # Count the number of image pixels in each accuracy category and export to a table
            create_pixels_per_category_table(
                    input_fclass_path = os.path.join(scratch_gdb, f'union_{damage_class}'),
                    input_image_path = os.path.join(prepared_test_images_gdb, fclass_name),
                    output_table_path = os.path.join(scratch_gdb, f'union_stats_{damage_class}'),
                    )

            # In the table, change the value of "ZONE_CODE" to 1 for each row to enable pivoting
            calculate_zone_code_field(
                    input_table_path = os.path.join(scratch_gdb, f'union_stats_{damage_class}'),
                    )

            # Pivot the table such that each accuracy category is a field with a total pixel count
            pivot_table(
                    input_table_path = os.path.join(scratch_gdb, f'union_stats_{damage_class}'),
                    output_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}'),
                    )

            # In the table, delete the "ZONE_CODE" field
            delete_zone_code_field(
                    input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}'),
                    )

            # In the table, create an "Image" field and assign the corresponding image name
            calculate_image_field(
                    input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}'),
                    image_name = fclass_name,
                    )

            # In the table, create a "Class" field and assign the corresponding damage class name
            calculate_class_field(
                    input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}'),
                    class_name = damage_class,
                    )

            # Verify that the table has a field for each accuracy category.
            # If not, create the field and assign a value of 0 (pixels).
            verify_accuracy_category_fields(
                    input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}'),
                    )

        # Update the progress indicators
        arcpy.SetProgressorPosition()
        fclasses_remaining -= 1
        arcpy.AddMessage(f'Calculated accuracy of {fclass_name}. {fclasses_remaining} predicted '
                         f'polygons feature classes remaining.')
        print(f'Calculated accuracy of {fclass_name}. {fclasses_remaining} predicted '
                         f'polygons feature classes remaining.')

    arcpy.SetProgressorLabel('Exporting accuracy results tables')
    print('Exporting accuracy results tables')

    for damage_class in damage_classes:
        # Create a list of accuracy tables for all images
        accuracy_tables = [os.path.join(scratch_gdb, f'accuracy_{damage_class}_{fclass_name}')
                           for fclass_name in predicted_polygons_fclasses]

        # Merge the tables into one accuracy table
        merge_accuracy_tables(
                input_table_paths = accuracy_tables,
                output_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Sum the TP, FP, and FN fields
        add_summary_row(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                class_name = damage_class,
                )

        # Create and calculate a "Union" field
        calculate_union_field(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Create and calculate a "Precision" field
        calculate_precision_field(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Create and calculate a "Recall" field
        calculate_recall_field(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Create and calculate an F1 score ("F1") field
        calculate_f1_field(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Create and calculate an intersection over union ("IoU") field
        calculate_iou_field(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                )

        # Export a final table with improved field settings
        improve_field_settings(
                input_table_path = os.path.join(scratch_gdb, f'accuracy_{damage_class}'),
                output_table_path = os.path.join(output_tables_gdb, f'Accuracy_{damage_class}'),
                )

    # Confirm processing is done
    arcpy.AddMessage(f'\nExported accuracy results tables to {output_tables_gdb}')
    print(f'\nExported accuracy results tables to {output_tables_gdb}')

    # Report which feature classes, if any, were skipped
    if len(skipped_fclasses) > 0:
        arcpy.AddMessage('\nThe following predicted polygons feature classes were skipped:')
        print('\nThe following predicted polygons feature classes were skipped:')
        for fclass in skipped_fclasses:
            arcpy.AddMessage(fclass)
            print(fclass)


# Calculate the accuracy of each predicted polygons feature class
if __name__ == '__main__':
    main()
