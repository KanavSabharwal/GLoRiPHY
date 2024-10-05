function io_write_iq(filename, data)
    % Ensure the file can be opened for writing
    fileID = fopen(filename, 'wb');
    if fileID == -1, error('Cannot open file: %s', filename); end
    
    interleavedData = zeros(2 * size(data, 2), 1, 'single');
    interleavedData(1:2:end) = real(data);
    interleavedData(2:2:end) = imag(data);
    
    % Write the entire interleaved array in one operation
    fwrite(fileID, interleavedData, 'single');
    
    fclose(fileID);
    disp(['File written: ', filename]);
end
