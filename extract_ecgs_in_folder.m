function extract_ecgs_in_folder(path)
    items = dir(path);
    for i=1:size(items,1)
        item = items(i);
        if strcmp(item.name, ".") || strcmp(item.name, "..")
            continue
        end
        whole_item_path = item.folder + "\" + item.name;
        if item.isdir == 1
            extract_ecgs_in_folder(whole_item_path);
        elseif endsWith(item.name, ".dcm") == 1
            disp("Extract ECG from file " + whole_item_path);
            info = dicominfo(whole_item_path);
            if isfield(info, 'CurveData_0')
                ecg = info.CurveData_0;
                images = dicomread(whole_item_path);  % WxHxCxN
                ecg = [size(images, 4); ecg];
                ecg_filename = whole_item_path + ".ecg.txt";
                dlmwrite(ecg_filename, ecg, ';');
            else
                disp("Error: DICOM file does not have CurveData_0 field.");
            end
        end
    end
end

