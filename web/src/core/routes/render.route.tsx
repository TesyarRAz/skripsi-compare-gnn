import { createBrowserRouter, RouterProvider } from "react-router";
import BaseRouter from "./base.route";

const browserRouter = createBrowserRouter(
    BaseRouter
)

const RenderRouter = () => {
    return (
        <RouterProvider
            router={browserRouter} 
        />
    )
}

export default RenderRouter