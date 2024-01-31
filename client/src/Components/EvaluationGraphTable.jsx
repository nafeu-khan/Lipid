import React from "react";
import AgGridTable from "./AgGridTable";

// TODO: get train-test loss graph and table from backend
// TODO: pass graph and table data from parent component
function EvaluationGraphTable({ graph = undefined, table = undefined }) {
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
      <div className="min-w-[500px]">
        <AgGridTable rowData={table ? table : []} />
      </div>
    </div>
  );
}

export default EvaluationGraphTable;
