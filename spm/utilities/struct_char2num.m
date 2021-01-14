function S = struct_char2num(S)
% loop over fields in a structure, and if some are type char, change to num
% (float or int). Change all "n/a" to NaNs.

fields= fieldnames(S);
for iField=1:length(fields)
    % this is an ugly way to look for 'n/a' in a char array
    if any(sum(ismember(S.(fields{iField}),'n/a'),2) == 3)
        % loop over rows because Matlab2016 doesn't know to apply strrep on
        % char arrays
        for iRow=1:size(S.(fields{iField}))
            S.(fields{iField})(iRow,:) = strrep(S.(fields{iField})(iRow,:),'n/a','NaN');
        end
        S.(fields{iField}) = str2num(S.(fields{iField}));
    end        
end