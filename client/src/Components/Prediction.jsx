import { Input, InputNumber, Radio, Space } from "antd";
import TextArea from "antd/es/input/TextArea";
import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import toast, { Toaster } from "react-hot-toast";

const { Group: RadioGroup, Button: RadioButton } = Radio;

/**
 * CompositionInput renders a flexible input form for lipid composition.
 * It displays a name input always and an optional percentage input.
 *
 * @param {Object} props - Component props
 * @param {number} props.id - Identifier for the composition, used in state management.
 * @param {boolean} props.showPercentage - Flag to show percentage input.
 * @param {string} props.name - Current name of the lipid composition.
 * @param {number} props.percentage - Current percentage of the lipid composition.
 * @param {Function} props.onCompositionChange - Callback to handle changes in composition data.
 */

const CompositionInput = ({
  id,
  showPercentage,
  name,
  percentage,
  onCompositionChange,
}) => (
  <div className="flex items-center gap-4">
    <div className="w-full">
      <label className="text-gray-800">
        Lipid Composition Name{showPercentage ? `-${id}` : ""}
      </label>
      <Input
        value={name}
        onChange={(e) =>
          onCompositionChange(`comp${id}`, "name", e.target.value)
        }
      />
    </div>
    {showPercentage && (
      <div>
        <label className="text-gray-800">Percentage</label>
        <Input
          value={`${percentage}`}
          onChange={(e) =>
            onCompositionChange(
              `comp${id}`,
              "percentage",
              parseInt(e.target.value) || 0
            )
          }
          type="number"
          suffix="%"
        />
      </div>
    )}
  </div>
);

/**
 * BeadsBondsInput provides an input mechanism for either uploading a file or entering data manually.
 * It switches between a file uploader and a text area based on the selected input type.
 *
 * @param {Object} props - Component props
 * @param {string} props.label - Label for the input section.
 * @param {string} props.inputType - Current selected input type ('upload' or 'custom').
 * @param {Function} props.setInputType - Callback to change the input type.
 */

const BeadsBondsInput = ({
  label,
  inputType,
  setInputType,
  value,
  handleFileChange,
  handleTextChange,
}) => (
  <div className="mt-4">
    <label className="text-gray-800">{label}</label>
    <div className="flex mt-2 gap-4 items-center">
      <RadioGroup
        value={inputType}
        onChange={(e) => setInputType(e.target.value)}
        buttonStyle="solid"
      >
        <Space direction="vertical">
          <RadioButton value="upload">Upload</RadioButton>
          <RadioButton value="custom">Custom</RadioButton>
        </Space>
      </RadioGroup>
      {inputType === "upload" ? (
        <FileUploader
          name="file"
          handleChange={(file) => {
            handleFileChange(file);
          }}
          classes="dndFile"
        />
      ) : (
        <TextArea
          placeholder="Input Data (Separated by comma)"
          value={value.text}
          onChange={(e) => handleTextChange(e.target.value)}
        />
      )}
    </div>
  </div>
);

/**
 * Prediction is the main component that manages the state and logic of the application.
 * It handles the rendering of composition inputs, file uploads, and other data fields.
 */

function Prediction() {
  const [type, setType] = useState("single");
  const [data, setData] = useState(initialDataState());
  const [compositions, setCompositions] = useState(initialCompositionState());
  const [adjacencyInputType, setAdjacencyInputType] = useState("upload");
  const [nodeFeatureInputType, setNodeFeatureInputType] = useState("upload");

  const [adjacencyInput, setAdjacencyInput] = useState({
    file: null,
    text: "",
  });
  const [nodeFeatureInput, setNodeFeatureInput] = useState({
    file: null,
    text: "",
  });

  const handleFileChange = (file, type) => {
    if (type === "adjacency") {
      setAdjacencyInput((prev) => ({ ...prev, file }));
      
    } else if (type === "nodeFeature") {
      setNodeFeatureInput((prev) => ({ ...prev, file }));
    }
  };

  const handleTextInputChange = (text, type) => {
    if (type === "adjacency") {
      setAdjacencyInput((prev) => ({ ...prev, text }));
    } else if (type === "nodeFeature") {
      setNodeFeatureInput((prev) => ({ ...prev, text }));
    }
  };

  const handleTypeChange = (newType) => {
    setType(newType);
    setCompositions(compositionStateOnTypeChange(newType));
  };

  const handleCompositionChange = (id, field, value) => {
    const updatedCompositions = getUpdatedCompositions(
      compositions,
      id,
      field,
      value,
      type
    );
    if (updatedCompositions) {
      setCompositions(updatedCompositions);
    }
  };

  const handleInputChange = (e, key) => {
    setData({ ...data, [key]: e });
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    
    // Add file and text data
    if (adjacencyInput.file) {
      formData.append("adjacencyFile", adjacencyInput.file);
    }
    formData.append("adjacencyText", adjacencyInput.text);

    if (nodeFeatureInput.file) {
      formData.append("nodeFeatureFile", nodeFeatureInput.file);
    }
    formData.append("nodeFeatureText", nodeFeatureInput.text);

    // Add other data fields
    formData.append("type", type);
    formData.append("compositions", JSON.stringify(compositions));
    formData.append("data", JSON.stringify(data));

    // Send the request
    try {
      console.log(formData.get("adjacencyFile"));
      const response = await fetch("http://localhost:8000/test/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log(result);
      // Handle result here
    } catch (error) {
      console.error("There was a problem with the fetch operation:", error);
    }
  };

  return (
    <div className="w-full h-full p-4">
      <Toaster />
      <div className="text-center mb-8">
        <RadioGroup
          name="radiogroup"
          size="large"
          defaultValue={type}
          onChange={(e) => handleTypeChange(e.target.value)}
          buttonStyle="solid"
        >
          <RadioButton value={"single"}>Single Composition</RadioButton>
          <RadioButton value={"multiple"}>Multiple Composition</RadioButton>
        </RadioGroup>
      </div>
      {type === "single" ? (
        <CompositionInput
          id={1}
          name={compositions.comp1.name}
          percentage={compositions.comp1.percentage}
          onCompositionChange={handleCompositionChange}
        />
      ) : (
        <>
          <CompositionInput
            id={1}
            showPercentage
            name={compositions.comp1.name}
            percentage={compositions.comp1.percentage}
            onCompositionChange={handleCompositionChange}
          />
          <CompositionInput
            id={2}
            showPercentage
            name={compositions.comp2.name}
            percentage={compositions.comp2.percentage}
            onCompositionChange={handleCompositionChange}
          />
        </>
      )}
      <BeadsBondsInput
        label="Beads-Bonds Structure (Adjacency Matrix)"
        inputType={adjacencyInputType}
        setInputType={setAdjacencyInputType}
        value={adjacencyInput}
        handleFileChange={(file) => handleFileChange(file, "adjacency")}
        handleTextChange={(text) => handleTextInputChange(text, "adjacency")}
      />
      <BeadsBondsInput
        label="Beads-Bonds Structure (Node Feature Matrix)"
        inputType={nodeFeatureInputType}
        setInputType={setNodeFeatureInputType}
        value={nodeFeatureInput}
        handleFileChange={(file) => handleFileChange(file, "nodeFeature")}
        handleTextChange={(text) => handleTextInputChange(text, "nodeFeature")}
      />
      <div className="grid grid-cols-2 gap-4 mt-6">
        {Object.keys(data).map((key) => (
          <div key={key}>
            <label htmlFor="" className="text-gray-800">
              {key.replace(/([A-Z])/g, " $1").trim()}
            </label>
            <InputNumber
              size="large"
              className="mt-1 w-full"
              value={data[key]}
              onChange={(e) => handleInputChange(e, key)}
            />
          </div>
        ))}
      </div>

      <div className="w-full mt-4 text-right">
        <button
          className="bg-blue-500 p-2 px-6 shadow rounded tracking-wider text-white font-medium"
          onClick={handleSubmit}
        >
          Predict
        </button>
      </div>
      <div className="my-4 text-2xl gap-4 mt-8 flex font-mono items-center justify-center">
        <h1 className="text-gray-800">
          Prediction for{" "}
          <span className="text-gray-900 font-bold tracking-wide">POPC</span>{" "}
          is:{" "}
        </h1>
        <p className="bg-violet-500 text-white font-bold p-2 px-4 rounded">
          20.30
        </p>
      </div>
    </div>
  );
}

/**
 * Returns the initial state for the data fields in the Prediction component.
 * It initializes all fields with empty strings.
 *
 * @returns {Object} The initial data state object.
 */

const initialDataState = () => ({
  "Number of Water": "",
  Salt: "",
  Temperature: "",
  Pressure: "",
  "Number of Lipid Per Layer": "",
  "Membrane Thickness": "",
  "Kappa KT(q^-4 + b)": "",
  "Kappa Binning (KT)": "",
  "Kappa Gamma / Binning": "",
  "Kappa BW DCF": "",
  "Kappa RSF": "",
});

/**
 * Returns the initial state for the lipid compositions.
 * Sets up the initial names and percentages for compositions.
 *
 * @returns {Object} The initial composition state object.
 */

const initialCompositionState = () => ({
  comp1: { name: "", percentage: 100 },
  comp2: { name: "", percentage: 0 },
});

/**
 * Determines the state of compositions based on the selected type ('single' or 'multiple').
 * In 'single' mode, it sets comp1 percentage to 100%, and in 'multiple' mode, it sets both to 0%.
 *
 * @param {string} newType - The selected composition type.
 * @returns {Object} The updated composition state object.
 */

const compositionStateOnTypeChange = (newType) =>
  newType === "single"
    ? { comp1: { name: "", percentage: 100 } }
    : {
        comp1: { name: "", percentage: 0 },
        comp2: { name: "", percentage: 0 },
      };

/**
 * Calculates and returns the updated state for compositions.
 * In 'multiple' mode, it validates the total percentage does not exceed 100%.
 * Returns null if validation fails.
 *
 * @param {Object} compositions - The current compositions state.
 * @param {string} id - The composition identifier being updated.
 * @param {string} field - The field in the composition being updated ('name' or 'percentage').
 * @param {number|string} value - The new value for the field.
 * @param {string} type - The current selected type ('single' or 'multiple').
 * @returns {Object|null} The updated compositions state object or null if validation fails.
 */

const getUpdatedCompositions = (compositions, id, field, value, type) => {
  if (field === "percentage" && type === "multiple") {
    const totalPercentage =
      id === "comp1"
        ? value + compositions.comp2.percentage
        : compositions.comp1.percentage + value;
    if (totalPercentage > 100) {
      toast.error("Total percentage cannot exceed 100%");
      return null;
    }
  }
  return {
    ...compositions,
    [id]: { ...compositions[id], [field]: value },
  };
};

export default Prediction;
