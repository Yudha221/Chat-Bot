prompt_template = """
Gunakan informasi berikut untuk menjawab pertanyaan dari pengguna.
Jika Anda tidak mengetahui jawabannya, cukup katakan bahwa Anda tidak tahu — jangan mencoba mengarang jawaban.

Konteks: {context}
Pertanyaan: {question}

Berikan jawaban yang bermanfaat dalam bahasa Indonesia saja, meskipun konteks mengandung bahasa Inggris.
Jangan tambahkan apapun selain jawaban.
Jawaban:
"""
