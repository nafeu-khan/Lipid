import { AppBar, Dialog, IconButton, Toolbar, Typography } from "@mui/material";
import React, { useState } from "react";

import CloseIcon from "@mui/icons-material/Close";
import { useSelector } from "react-redux";
import ChartComponent from "./ChartComponent";

function ActualPredicted() {
  const [open, setIsOpen] = useState(false);
  const data = useSelector((state) => state.lipid.data);
  const lipid = useSelector((state) => state.lipid.lipid);
  if (Object.keys(data).length == 0) return;
  let actual = undefined,
    predicted = undefined;
  if (data) {
    if (data.actual) {
      actual = data.actual[lipid[0].name];
    }
    if (data.predicted) {
      predicted = data.predicted[lipid[0].name];
    }
  }
  console.log({ actual, predicted });

  return (
    <div className="absolute top-1.5 right-1 z-50">
      <button
        className="p-2 bg-violet-500 shadow px-3 rounded text-sm text-white"
        onClick={() => setIsOpen(true)}
      >
        Actual vs Predicted
      </button>
      <Dialog open={open} fullScreen onClose={() => setIsOpen(false)}>
        <AppBar sx={{ position: "relative" }} className="!bg-violet-500">
          <Toolbar>
            <IconButton
              edge="start"
              color="inherit"
              onClick={() => setIsOpen(false)}
              aria-label="close"
            >
              <CloseIcon />
            </IconButton>
            <Typography sx={{ ml: 2, flex: 1 }} variant="h6" component="div">
              Actual vs Predicted
            </Typography>
          </Toolbar>
        </AppBar>
        <div className="p-4 grid grid-cols-2 h-full">
          <div className=" flex flex-col h-full">
            <h1 className="text-center font-medium text-xl">Actual</h1>
            <div className="w-full h-full">
              {/* TODO: Change [actual] if working with multiple lipid component. Including predicted */}
              <ChartComponent graph_data={actual && [actual]} id="2" />
            </div>
          </div>
          <div className="border-l-4 border-gray-200 rounded-xl flex flex-col h-full">
            <h1 className="text-center font-medium text-xl">Predicted</h1>
            <div className="w-full h-full">
              <ChartComponent graph_data={predicted && [predicted]} id="3" />
            </div>
          </div>
        </div>
      </Dialog>
    </div>
  );
}

export default ActualPredicted;
