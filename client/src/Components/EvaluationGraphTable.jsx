import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import AgGridTable from "./AgGridTable";

// TODO: get train-test loss graph and table from backend
// TODO: pass graph and table data from parent component
function EvaluationGraphTable({ operationID }) {
  const { data, loading } = useSelector((state) => state.evaluation);
  const [graph, setGraph] = useState("");
  const [table, setTable] = useState(undefined);

  useEffect(() => {
    if (!data) return;
    if (operationID === "loss") {
      setGraph(data.loss.graph);
      setTable(JSON.parse(data.loss.table));
    } else if (operationID === "actualvspred") {
      setGraph(data.actualvspred.graph);
      setTable(JSON.parse(data.actualvspred.table));
    }
  }, [data, operationID]);

  if (loading)
    return (
      <div className="w-full h-full flex items-center justify-center">
        <h1 className="text-3xl font-bold">Fetching Data...</h1>
      </div>
    );

  if(operationID === 'r2' && data) {
    return <div className="w-full h-full grid place-items-center">
      <h1 className="text-3xl">R-squared Score: {data['R-squared']}</h1>
    </div>
  }

  return (
    <div className="flex w-full gap-4">
      <div className="w-full flex items-center">
        {graph ? (
          <img
            src={`data:image/png;base64,${graph}`}
            alt="Loss Graph"
            className="w-full "
          />
        ) : (
          <h2 className="text-center w-full font-medium text-xl">
            No Graph Found
          </h2>
        )}
      </div>
      <div className="min-w-[400px]">
        <AgGridTable rowData={table ? table : []} />
      </div>
    </div>
  );
}

export default EvaluationGraphTable;
