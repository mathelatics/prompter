#!/usr/bin/env python3
"""make_test_pptx.py — build a minimal valid .pptx containing presenter notes.

Used by cPrompter's test suite (test_pptx_load) as a fixture generator so the
import path is exercised against a real ZIP/OOXML file, not hand-fed strings.
Writes: [Content_Types].xml, _rels/.rels, ppt/presentation.xml,
ppt/_rels/presentation.xml.rels, ppt/slides/slideN.xml(+rels),
ppt/notesSlides/notesSlideN.xml(+rels). Notes text supplied per slide.
Usage: python3 make_test_pptx.py out.pptx
"""
import sys, zipfile

NOTES = {
    1: "Opening remarks\n& thanks to the sponsors",
    2: "Closing line\n<final> reminder to subscribe",
}

CT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
'<Default Extension="xml" ContentType="application/xml"/>'
'<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
'<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
'<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
'<Override PartName="/ppt/notesSlides/notesSlide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
'<Override PartName="/ppt/notesSlides/notesSlide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
'</Types>')

ROOT_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
'</Relationships>')

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

PRES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
'<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
'<p:sldIdLst><p:sldId id="256" r:id="rId2" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
'<p:sldId id="257" r:id="rId3" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></p:sldIdLst>'
'</p:presentation>')

PRES_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>'
'<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>'
'</Relationships>')

def slide_xml(n):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    '<p:cSld><p:spTree/></p:cSld></p:sld>')

def notes_xml(n):
    # NOTES[n] uses literal "\n" marker to split paragraphs; each paragraph is
    # one <a:p> with a single run, so extracted text must come back exactly.
    parts = NOTES[n].split('\n')
    body = ''.join('<a:p><a:r><a:t>' + esc(p) + '</a:t></a:r></a:p>' for p in parts)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<p:notesSlide xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
    '<p:cSld><p:spTree><p:sp><p:txBody>' + body + '</p:txBody></p:sp></p:spTree></p:cSld>'
    '</p:notesSlide>')

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "test.pptx"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("ppt/presentation.xml", PRES)
        z.writestr("ppt/_rels/presentation.xml.rels", PRES_RELS)
        for n in (1, 2):
            z.writestr(f"ppt/slides/slide{n}.xml", slide_xml(n))
            z.writestr(f"ppt/notesSlides/notesSlide{n}.xml", notes_xml(n))
    print(out)

if __name__ == "__main__":
    main()
