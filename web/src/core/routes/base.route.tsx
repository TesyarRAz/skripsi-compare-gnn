import type { RouteObject } from "react-router";
import BaseLayout from "../layout/base.layout";
import BaseTspRoutes from "../../modules/tsp/routes/base.route";

const BaseRouter: RouteObject[] = [
    {
        path: '/',
        element: <BaseLayout />,
        children: [
            ...BaseTspRoutes
        ]
    }
]

export default BaseRouter