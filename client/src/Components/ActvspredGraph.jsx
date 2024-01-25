import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import AgGridTable from "./AgGridTable";

function ActvspredGraph() {
  // Assuming results_json is available in your Redux state
  const resultsJson = useSelector((state) => state.lipid.data.results_json);
  const [rowData, setRowData] = useState([]);

  useEffect(() => {
    // Check if resultsJson is available
    if (!resultsJson) return;

    // Parse the JSON string to an object
    const resultsData = JSON.parse(resultsJson);

    // Set the parsed data as row data for the table
    setRowData(resultsData);
  }, [resultsJson]);

  return <AgGridTable rowData={rowData} />;
}

export default ActvspredGraph;
