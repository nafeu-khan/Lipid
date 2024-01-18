import { configureStore } from "@reduxjs/toolkit";
import lipidReducer from "./Slices/LipidSlice";

export const store = configureStore({
  reducer: {
    lipid: lipidReducer,
  },
});
