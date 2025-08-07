import { Container, Typography } from '@mui/material'
import { Outlet } from 'react-router'

const BaseLayout = () => {
  return (
    <Container maxWidth="xl" className='flex flex-col min-h-screen h-screen'>
      <header className="bg-gray-800 text-white p-4">
        <Typography variant="h5">
          BaseLayout
        </Typography>
      </header>
      <main className="flex-1 flex p-4">
        <Outlet />
      </main>
      <footer className="bg-gray-800 text-white p-4 text-center">
        &copy; 2025 TesyarRAz
      </footer>
    </Container>
  )
}

export default BaseLayout