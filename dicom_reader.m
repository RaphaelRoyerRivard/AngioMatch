path = "C:\Users\root\Data\new_data\P28\LCA_0_0.dcm";
info = dicominfo(path);
ecg = info.CurveData_0;
images = dicomread(path);
[max_value, index] = max(ecg);
frame = index / size(ecg,1) * size(images, 4)