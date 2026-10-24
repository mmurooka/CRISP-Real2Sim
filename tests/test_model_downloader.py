import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'setups'))
from download_interactvlm_model import download, MODELS


class Response:
    def __init__(self,content,status=200,headers=None):
        self.content=content
        self.status_code=status
        self.headers={'Content-Length':str(len(content)),**(headers or {})}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self): pass
    def iter_content(self,**kwargs): yield self.content


class DownloadTests(unittest.TestCase):
    def setUp(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as a: a.writestr('model.bin',b'example weights')
        self.archive=stream.getvalue()

    def test_official_model_selection_and_zip_validation(self):
        with tempfile.TemporaryDirectory() as tmp, patch('download_interactvlm_model.requests.post',return_value=Response(self.archive)) as post:
            download(tmp,'example_user','example_password','smpl')
            self.assertIn('domain=smpl&sfile=SMPL_python_v.1.1.0.zip',post.call_args.args[0])
            self.assertEqual((Path(tmp)/MODELS['smpl'][1]).read_bytes(),self.archive)
            with self.assertRaises(FileExistsError): download(tmp,'user','password','smpl')

    def test_login_page_does_not_replace_partial_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial=Path(tmp)/(MODELS['smplx'][1]+'.part')
            partial.write_bytes(self.archive[:12])
            with patch('download_interactvlm_model.requests.post',return_value=Response(b'<html>login</html>',headers={'Content-Type':'text/html'})):
                with self.assertRaises(RuntimeError): download(tmp,'user','password','smplx')
            self.assertEqual(partial.read_bytes(),self.archive[:12])

    def test_valid_resume_appends_only_requested_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            name=MODELS['interactvlm'][1]
            (Path(tmp)/(name+'.part')).write_bytes(self.archive[:12])
            response=Response(self.archive[12:],206,{'Content-Range':f'bytes 12-{len(self.archive)-1}/{len(self.archive)}'})
            with patch('download_interactvlm_model.requests.post',return_value=response) as post:
                download(tmp,'user','password')
            self.assertEqual(post.call_args.kwargs['headers']['Range'],'bytes=12-')
            self.assertEqual((Path(tmp)/name).read_bytes(),self.archive)


if __name__=='__main__': unittest.main()
