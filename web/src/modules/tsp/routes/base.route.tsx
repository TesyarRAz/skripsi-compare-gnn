import type { RouteObject } from 'react-router'

const BaseTspRoutes: RouteObject[] = [
    {
        path: '/',
        lazy: async () => {
            const Dashboard = await import('./../pages/dashboard')

            return {
                Component: Dashboard.default,
            }
        },
        handle: {
            title: 'Dashboard',
            trackCode: 'dashboard',
        }
    }
]

export default BaseTspRoutes