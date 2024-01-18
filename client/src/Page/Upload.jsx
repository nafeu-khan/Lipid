import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import LoadingButton from "@mui/lab/LoadingButton";
import { Button } from "@mui/material";
import { styled } from "@mui/material/styles";
import * as React from "react";
import AgGridTable from "../Components/AgGridTable";
import bendingFile from "../bending-modulus.xlsx";
import molecules from "../molecules.csv";
import { parseCsv, parseExcel } from "../utility";

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

function Upload() {
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState([]);

  const handleFileChange = async (evt) => {
    setLoading(true);
    const file = evt.target.files[0];
    const mainData = await parseExcel(bendingFile);

    let fileData;
    if (file.type.toLowerCase().includes("csv")) {
      fileData = await parseCsv(file);
    } else {
      fileData = await parseExcel(file);
    }
    // setData(fileData);
    const temp_key = Object.keys(fileData[0]);
    const temp_data = fileData.map((val) => "100% " + val[temp_key]);

    let matchingValues = [];
    const usedTempData = new Set();

    mainData.forEach((item) => {
      const lipidValue = item["Lipid composition (molar)"];
      if (temp_data.includes(lipidValue) && !usedTempData.has(lipidValue)) {
        matchingValues.push(item);
        usedTempData.add(lipidValue);
      }
    });

    matchingValues = matchingValues.map(
      ({ Index, Path, Barostat, Thermostat, "Salt, M": salt, ...rest }) => rest
    );

    const moleculesData = await parseCsv(molecules);
    console.log(moleculesData);

    // Assuming matchingValues and moleculesData are already defined and populated
    for (let i = 0; i < matchingValues.length; i++) {
      // Find the corresponding entry in moleculesData
      const correspondingEntry = moleculesData.find(
        (molecule) =>
          molecule["Lipid composition (molar)"] ===
          matchingValues[i]["Lipid composition (molar)"]
      );
      // If a matching entry is found, parse and add the 'Nodes_features' values
      if (correspondingEntry) {
        matchingValues[i] = {
          node_features: correspondingEntry.node_features,
          edge: correspondingEntry.edge,
          ...matchingValues[i],
        };
      }
    }

    // Now matchingValues includes the parsed nodes and features from Nodes_features where the Lipid composition (molar) matches

    setData(matchingValues);

    setLoading(false);
  };
  return (
    <div className="min-h-screen grid content-center px-8 w-full mx-auto">
      <div className="text-center space-x-4">
        <LoadingButton
          loading={loading}
          loadingIndicator="Uploading..."
          component={"label"}
          variant="contained"
          startIcon={<CloudUploadIcon />}
          onChange={handleFileChange}
        >
          Upload file
          <VisuallyHiddenInput type="file" />
        </LoadingButton>
        {!!data.length && (
          <Button
            color="error"
            onClick={() => {
              setData([]);
              window.location.reload();
            }}
          >
            Reset
          </Button>
        )}
      </div>
      {data && data.length > 0 && (
        <div className="w-full mt-10">
          <AgGridTable rowData={data} />
        </div>
      )}
    </div>
  );
}

export default Upload;
