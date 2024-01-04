import PyPDF2
filenames=['封面.pdf','测试报告.pdf']
merger=PyPDF2.PdfMerger()
for filename in filenames:
    merger.append(PyPDF2.PdfReader(filename))
merger.write('测试报告_3210105952_艾子翔.pdf')
