function output = get_data(filename, embeddings)
    table = readtable(filename);
    cells = table2array(table);
    cells = cells(:,1:2);
    expression = '''\S+''';
    output = {size(cells,1), size(cells,2)};

    for i = 1:size(cells,1)
        string = cell2mat(cells(i,1));
        matchStr = regexp(string, expression, 'match');
        embedded_string = zeros(size(matchStr,2), 64);
        for j = 1:size(matchStr,2)
            word = cell2mat(matchStr(j));
            word = convertCharsToStrings(word(1,2:size(word,2)-1));
            embedded_string(j,:) = embeddings(word);
        end

        positional_encoding = get_positional_encoding(size(matchStr,2), 64, 10000);
        embedded_string = embedded_string + positional_encoding;

        tags = cell2mat(cells(i,2));
        tags = tags(2:size(tags,2)-1);
        tags = str2num(tags);
        for j = 1:size(tags,2)
            if ((tags(j) >= 21) && (tags(j) <= 25))||(tags(j) == 28)||(tags(j) == 29)
                tags(j) = 1;
            elseif (tags(j) >= 37) && (tags(j) <= 42)
                tags(j) = 2;
            elseif ((tags(j) >= 16) && (tags(j) <= 18))||((tags(j) >= 30) && (tags(j) <= 32))
                tags(j) = 3;
            else
                tags(j) = 4;
            end
        end

        output(i,:) = {embedded_string, tags};
        
    end

end