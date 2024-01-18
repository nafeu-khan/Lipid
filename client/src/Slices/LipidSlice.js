import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { graph_data } from "../utility";

const graphData = JSON.parse(graph_data);
const URL = "http://127.0.0.1:8000";

export const getMoleculeStructure = createAsyncThunk(
  "lipid/getMoleculeStructure",
  async (mol_name) => {
    const res = await fetch(`${URL}/edge_pred/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        mol_name,
      }),
    });
    const data = await res.json();
    return { data, mol_name };
  }
);

const initialState = {
  lipid: [],
  type: "single",
  operationID: "0",
  loading: false,
  data: {},
};

export const lipidSlice = createSlice({
  name: "lipid",
  initialState,
  reducers: {
    changeActiveLipid: (state, { payload }) => {
      state.lipid = payload;
    },
    changeNumOfComp: (state, { payload }) => {
      state.type = payload;
    },
    changeOperationID: (state, { payload }) => {
      state.operationID = payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(getMoleculeStructure.fulfilled, (state, { payload }) => {
      const { linkWith, ...rest } = payload.data;
      state.data = {
        ...state.data,
        actual: { [payload.mol_name]: graphData[payload.mol_name] },
        predicted: { [payload.mol_name]: rest },
      };
      state.loading = false;
    });
    builder.addCase(getMoleculeStructure.pending, (state) => {
      state.loading = true;
    });
    builder.addCase(getMoleculeStructure.rejected, (state) => {
      state.loading = false;
    });
  },
});

// Action creators are generated for each case reducer function
export const { changeActiveLipid, changeNumOfComp, changeOperationID } =
  lipidSlice.actions;

export default lipidSlice.reducer;
