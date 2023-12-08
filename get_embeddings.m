function embeddings = get_embeddings(filename)
    embeddings_table = readtable(filename);
    embeddings_mat = table2array(embeddings_table);
    
    words_cell = embeddings_mat(:,1);
    vecs_cell = embeddings_mat(:,2);
    embeddings = containers.Map;
    
    for i = 1:size(embeddings_mat,1)
        word = convertCharsToStrings(cell2mat(words_cell(i)));
        str = cell2mat(vecs_cell(i));
        str = replace(convertCharsToStrings(str(3:size(str,2) - 1)), newline, '');
        embeddings(word) = str2num(str);
    end
end