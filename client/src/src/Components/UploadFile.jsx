import React from "react";
import { FileUploader } from "react-drag-drop-files";

function UploadFile() {
  return (
    <div className="mt-6">
      <p className="font-medium text-gray-800/80 underline text-center mb-4">
        Upload File
      </p>
      <FileUploader
        name="file"
        types={["csv", "xlsx"]}
        onDrop={(e) => console.log(e)}
        onChange={(e) => console.log(e)}
        classes="dndFile"
      />
    </div>
  );
}

export default UploadFile;
