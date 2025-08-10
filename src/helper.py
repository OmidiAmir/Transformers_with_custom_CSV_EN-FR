

def seq_len_with_specials(text, tok):
    return 2 + len(tok.encode(str(text)).ids)  # BOS + EOS