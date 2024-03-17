import { Radio, Space } from "antd";
import TextArea from "antd/es/input/TextArea";
import React from "react";
import { FileUploader } from "react-drag-drop-files";

const { Group: RadioGroup, Button: RadioButton } = Radio;

const BeadsBondsInput = ({
  label,
  inputType,
  inputKey, // Added to correctly identify which input's state to update
  file,
  text,
  setInputType,
  handleFileChange,
  handleTextChange,
}) => (
  <div className="mt-4">
    <label className="text-gray-800 font-semibold">{label}</label>
    <div className="flex mt-2 gap-4 items-center">
      <RadioGroup
        value={inputType}
        onChange={(e) => setInputType(inputKey, e.target.value)} // Adjusted to include inputKey
        buttonStyle="solid"
      >
        <Space direction="vertical">
          <RadioButton value="upload">Upload</RadioButton>
          <RadioButton value="custom">Custom</RadioButton>
        </Space>
      </RadioGroup>
      {inputType === "upload" ? (
        <FileUploader
          handleChange={(file) => handleFileChange(inputKey, file)} // Adjusted to include inputKey
          name={inputKey} // This should be unique per input
          types={["TXT", "CSV"]}
        />
      ) : (
        <TextArea
          placeholder="Input Data (Separated by comma)"
          value={text}
          onChange={(e) => handleTextChange(inputKey, e.target.value)} // Adjusted to include inputKey
        />
      )}
    </div>
  </div>
);

export default BeadsBondsInput;
