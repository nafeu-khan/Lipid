import { Radio } from "antd";
import React, { useState } from "react";
import { useSelector } from "react-redux";
import AgGridTable from "./AgGridTable";

const PredictionKappa = () => {
  const { data, lipid } = useSelector((state) => state.lipid);
  const [selectedValue, setSelectedValue] = useState("test/train");
  console.log(data.actualvspred);

  const handleChange = (value) => {
    setSelectedValue(value);
  };

  return (
    <div className="w-full h-full flex flex-col items-center p-2 ">
      <div className="flex items-center gap-2 mb-4 mt-2">
        <Radio.Group
          value={selectedValue}
          size="large"
          buttonStyle="solid"
          onChange={(e) => handleChange(e.target.value)}
        >
          <Radio.Button value={"test/train"}>Test/Train Loss</Radio.Button>
          <Radio.Button value={"r2"}>R-Square</Radio.Button>
          <Radio.Button value={"actualvspred"}>
            Actual vs Predicted
          </Radio.Button>
        </Radio.Group>
      </div>
      <div className="bg-violet-500 mt-2 p-2 px-4 rounded shadow ">
        <h1 className=" text-gray-100 text-lg font-mono">
          Lipid Name:{" "}
          <span className="text-white font-semibold">{lipid[0].name}</span>
        </h1>
        <h1 className=" text-gray-100 text-lg font-mono">
          Prediction Value:{" "}
          <span className="text-white font-semibold">
            {data && data.pred ? data.pred.toFixed(3) : "null"}
          </span>
        </h1>
      </div>

      <div className="w-full mt-12">
        {selectedValue === "test/train" && (
          <div className="flex w-full gap-4">
            <div className="w-full flex items-center">
              {data.loss && data.loss.graph ? (
                <img
                  src={`data:image/png;base64,${data.loss.graph}`}
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
              <AgGridTable
                rowData={data.loss && data.loss.table ? data.loss.table : []}
              />
            </div>
          </div>
        )}

        {selectedValue === "r2" && (
          <div className="flex w-full gap-4">
            <div className="w-full flex items-center">
              {data.r2 && data.r2.graph ? (
                <img
                  src={`data:image/png;base64,${data.r2.graph}`}
                  alt="R2 Graph"
                  className="w-full h-full"
                />
              ) : (
                <h2 className="text-center w-full font-medium text-xl">
                  No Graph Found
                </h2>
              )}
            </div>
            <div className="min-w-[400px]">
              <AgGridTable
                rowData={data.r2 && data.r2.table ? data.r2.table : []}
              />
            </div>
          </div>
        )}

        {selectedValue === "actualvspred" && (
          <div className="mx-auto max-w-[350px]">
            <AgGridTable rowData={data.actualvspred ? data.actualvspred : []} />
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionKappa;
