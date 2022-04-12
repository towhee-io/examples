# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, status, HTTPException
from io import BytesIO
import towhee
import utils
from starlette.responses import StreamingResponse

app = FastAPI()

@app.post("/api/trans")
async def transform_api(file: UploadFile = File(...), model_name: str = Form(...)):
    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not extension:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=f'File {file.filename} should be jpg, jpeg or png')

    if model_name.lower() not in ['celeba', 'facepaintv1', 'facepaitv2', 'hayao', 'paprika', 'shinkai']:
        return f"Specified Model: {model_name} Name Does not exist"
    
    input_image = utils.read_image(file.file.read())
    file.file.close()
    output_image = utils.translate_image(input_image, model_name)
    filtered_image = BytesIO()
    output_image.save(filtered_image, "PNG")
    filtered_image.seek(0)

    return StreamingResponse(filtered_image, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app)